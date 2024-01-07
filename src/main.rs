use std::collections::BTreeMap;
use std::io::Write;
use std::ops::Range;
use std::os::fd::AsRawFd;
use std::thread;
use std::{fs, ptr};

use anyhow::{anyhow, Result};

const EMPTY_NAME: [u8; 100] = [0; 100];
const NUM_THREADS: usize = 10; // Machine specific
static mut TABLE: [[(Option<[u8; 100]>, Measure); 10_000]; 10] =
    [[(None, Measure::ZERO); 10_000]; 10];
static mut POPULATED_STATIONS: [(usize, [usize; 10_000]); 10] = [(0, [usize::MAX; 10_000]); 10];

static mut PAGE_SIZE: usize = 0;

fn main() -> Result<()> {
    let file = fs::File::open("measurements.txt")?;
    let (ptr, len) = unsafe { map_file(&file)? };
    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
    unsafe {
        use libc::{sysconf, _SC_PAGESIZE};

        PAGE_SIZE = sysconf(_SC_PAGESIZE) as usize;
    }

    let splits = find_splits(slice);
    for s in splits.iter() {
        debug_assert!(
            slice[s.start - 1] == b'\n',
            "Expected newline, got {}",
            slice[s.start - 1]
        );
        debug_assert!(s.end == slice.len() || slice[s.end] == b'\n');
    }

    let map = thread::scope(|s| {
        let handles: Vec<_> = splits
            .into_iter()
            .enumerate()
            .map(|(tid, split)| {
                s.spawn(move || {
                    process(tid, &slice[split]);

                    tid
                })
            })
            .collect();

        let mut map = BTreeMap::new();
        for h in handles {
            let tid = h.join().expect("Threads to not panic");
            // SAFETY: No other thread can be accessing this
            let station_end = unsafe { POPULATED_STATIONS[tid].0 };

            for idx in 0..station_end {
                // SAFETY: No other thread can accessing this
                let idx = unsafe { POPULATED_STATIONS[tid].1[idx] };
                let (station, data) = unsafe { &TABLE[tid][idx] };
                let station = station.unwrap();
                let name = {
                    let end = station
                        .iter()
                        .enumerate()
                        .find_map(|(i, b)| (b == &0).then_some(i))
                        .unwrap();

                    // Guaranteed to be ASCII
                    unsafe { std::str::from_utf8_unchecked(&station[..end]) }
                };
                map.entry(name.to_owned())
                    .and_modify(|m: &mut Measure| m.merge(*data))
                    .or_insert_with(|| *data);
            }
        }

        map
    });

    let mut stdout = std::io::stdout().lock();
    stdout.write("{".as_bytes())?;

    let len = map.len();
    for (i, (key, data)) in map.iter().enumerate() {
        let mean = u32_to_f32(data.sum) / u32_to_f32(data.count);

        stdout.write_fmt(format_args!(
            "{}={:.1}/{:.1}/{:.1}",
            key,
            temp_to_f32(data.min),
            temp_to_f32(mean as u16),
            temp_to_f32(data.max),
        ))?;

        if i < len - 1 {
            stdout.write(", ".as_bytes())?;
        }
    }
    stdout.write("}".as_bytes())?;

    Ok(())
}

fn find_splits(slice: &[u8]) -> [Range<usize>; NUM_THREADS] {
    let total_len = slice.len();
    let expected_len = total_len / NUM_THREADS;
    let mut result = [0; NUM_THREADS].map(|x| x..x);

    let mut current_idx = 0;

    for i in 0..NUM_THREADS {
        let mut end = (current_idx + expected_len).min(slice.len());
        while end < total_len && slice[end] != b'\n' {
            end += 1;
        }
        result[i] = current_idx..end;
        current_idx = end + 1;
    }

    result
}

unsafe fn map_file(file: &fs::File) -> Result<(*const u8, usize)> {
    let meta = file.metadata()?;
    let len = meta.len() as usize;

    let ptr = libc::mmap(
        ptr::null_mut::<libc::c_void>(),
        len,
        libc::PROT_READ,
        libc::MAP_PRIVATE,
        file.as_raw_fd(),
        0,
    );

    if ptr == libc::MAP_FAILED {
        let error = std::io::Error::last_os_error();
        return Err(anyhow!("Failed to mmap file. {error}"));
    }

    Ok((ptr as *const u8, len))
}

fn process(thread_idx: usize, slice: &[u8]) {
    debug_assert!(slice[0] != b'\n');
    let mut current = slice;

    while current.len() > 0 {
        let split_idx = current
            .iter()
            .enumerate()
            .find(|(_, b)| b == &&b';')
            .map(|(i, _)| i)
            .unwrap();
        let line_end = current
            .iter()
            .enumerate()
            .find(|(_, b)| b == &&b'\n')
            .map(|(i, _)| i)
            .unwrap_or_else(|| current.len());

        let name = &current[..split_idx];
        let temp = &current[split_idx + 1..line_end];
        let parsed_temp = temp_from_bytes(temp);

        unsafe {
            // SAFETY: Only one thread uses this part of TABLE
            let idx = find_table_idx(
                &name,
                &mut TABLE[thread_idx],
                &mut POPULATED_STATIONS[thread_idx],
            );
            TABLE[thread_idx][idx].1.update(parsed_temp);
        };

        if (line_end + 1) >= current.len() {
            break;
        }

        current = &current[line_end + 1..];
    }
}

fn find_table_idx(
    name: &[u8],
    table: &mut [(Option<[u8; 100]>, Measure)],
    populated_stations: &mut (usize, [usize; 10_000]),
) -> usize {
    let key = {
        // Poor man's hash
        let mut key = [0; 8];
        for (i, b) in name.iter().take(8).enumerate() {
            key[8 - i - 1] = *b;
        }

        usize::from_be_bytes(key)
    };

    let mut idx = key % table.len();

    loop {
        let candidate = &mut table[idx];
        match candidate.0 {
            None => {
                // unoccupied
                let mut full_key = EMPTY_NAME;
                for (i, b) in name.iter().enumerate() {
                    full_key[i] = *b;
                }

                candidate.0 = Some(full_key);
                populated_stations.1[populated_stations.0] = idx;
                populated_stations.0 += 1;
                break;
            }

            Some(existing) => {
                if existing.iter().zip(name.iter()).all(|(a, b)| a == b) {
                    break;
                } else {
                    idx = (idx + 1) % table.len();
                }
            }
        }
    }

    idx
}

// -99.9 through 99.9 always with one fractional digit, stored as a u16 in 0.1 increments.
// 0     =  -99.9
// 1     =  -99.8
// ..
// 999   = 0
// 1000  = 0.1
// 1998  = 99.9
type Temp = u16;

fn temp_from_bytes(bytes: &[u8]) -> Temp {
    match bytes.len() {
        3 => {
            // y.f
            let f = (bytes[2] - b'0') as u16;
            let y = (bytes[0] - b'0') as u16;

            999 + y * 10 + f
        }
        4 if bytes[0] == b'-' => {
            // -y.f
            let f = (bytes[3] - b'0') as u16;
            let y = (bytes[1] - b'0') as u16;

            999 - (y * 10 + f)
        }
        4 => {
            // xy.f
            let f = (bytes[3] - b'0') as u16;
            let y = (bytes[1] - b'0') as u16;
            let x = (bytes[0] - b'0') as u16;

            999 + (x * 100 + y * 10 + f)
        }
        5 => {
            // -xy.f
            let f = (bytes[4] - b'0') as u16;
            let y = (bytes[2] - b'0') as u16;
            let x = (bytes[1] - b'0') as u16;

            999 - (x * 100 + y * 10 + f)
        }
        _ => unreachable!(),
    }
}

fn temp_to_f32(temp: Temp) -> f32 {
    (temp as f32 / 10.0) - 99.9
}

fn u32_to_f32(temp: u32) -> f32 {
    (temp as f32 / 10.0) - 99.9
}

#[derive(Clone, Copy)]
struct Measure {
    min: u16,
    max: u16,
    sum: u32, // Max 2^16 entries per station
    count: u32,
}

impl Measure {
    const ZERO: Self = Self {
        min: u16::MAX,
        max: 0,
        sum: 0,
        count: 0,
    };

    fn update(&mut self, value: u16) {
        self.min = self.min.min(value);
        self.max = self.max.max(value);
        self.sum += value as u32;
        self.count += 1;
    }

    fn merge(&mut self, other: Measure) {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
        self.sum += other.sum;
        self.count += other.count;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temp_from_bytes() {
        for i in 0..=1998 {
            let f = -99.9 + (i as f64) * 0.1;
            let as_str = format!("{f:.1}");
            let value = temp_from_bytes(as_str.as_bytes());

            let diff = (f - ((value as f64 / 10.0) - 99.9)).abs();
            assert!(
                diff < 0.01,
                "temp_from_bytes yields the wrong value for {as_str}. Diff is {diff}"
            );
        }
    }
}
