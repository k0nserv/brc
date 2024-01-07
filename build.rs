use std::env;
use std::fs;
use std::io::BufWriter;
use std::io::Seek;
use std::io::Write;
use std::path::PathBuf;

use anyhow::Result;

fn main() -> Result<()> {
    // let out: PathBuf = env::var("OUT_DIR")
    //     .unwrap()
    //     .try_into()
    //     .expect("OUT_DIR valid path");
    // let file = fs::OpenOptions::new()
    //     .create(true)
    //     .write(true)
    //     .truncate(true)
    //     .open(out.join("temp.bin"))?;
    // let mut buf_writer = BufWriter::with_capacity(151587073, file);

    // write_temperature_lookup_table(&mut buf_writer)?;

    println!("yes");

    Ok(())
}

fn write_temperature_lookup_table<W>(to: &mut W) -> Result<()>
where
    W: Write + Seek,
{
    // to.write("const TEMPS: [u16; 151587074] = [".as_bytes())?;

    let mut result = Vec::with_capacity(1998);

    // Ten digit
    for m in [true, false].into_iter() {
        // Ten digit
        for x in 0..10 {
            // Ones digits
            for y in 0..10 {
                // Fraction digit
                for f in 0..10 {
                    let idx = u64::from_be_bytes([
                        0x0,
                        0x0,
                        0x0,
                        0x0,
                        x,
                        y,
                        f,
                        if m { 0x01 } else { 0x0 },
                    ]);
                    let value = 0 + f as u16 + y as u16 * 10 + x as u16 * 100;
                    // Max: 0x2D00002E00
                    // Min: 0x0000002E00
                    // (Minus last)
                    // Max: 0x39392E002D
                    // Min: 0x00002E0000
                    // 0x2D39392E39
                    //
                    // 0 = 0x2D39392E39 - x
                    // 1 = 0x2D39392E38 - x
                    // 10 = 0x2D39382E30 - x

                    // Max: 0x39393901
                    // Min: 0x00000000
                    //
                    // Max: 0x09090901
                    // Min: 0x00000000
                    result.push((idx, value));
                }
            }
        }
    }

    let zero_bytes = 0_u16.to_be_bytes();
    for _ in 0..151587074 {
        to.write(&zero_bytes)?;
    }

    for (i, v) in result.into_iter() {
        to.seek(std::io::SeekFrom::Start(i * 2))?;
        to.write(&v.to_be_bytes())?;
    }

    Ok(())
}
