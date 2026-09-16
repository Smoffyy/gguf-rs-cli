use std::process::Command;

pub struct ClockLock {
    index: Option<u32>,
}

impl ClockLock {
    pub fn none() -> Self { Self { index: None } }

    pub fn engage(device_name: &str) -> Self {
        let Some(index) = find_smi_index(device_name) else {
            eprintln!("[gpu] nvidia-smi not found or GPU isn't NVIDIA — skipping clock lock.");
            return Self::none();
        };
        let Some(max_clock) = query_max_clock(index) else {
            eprintln!("[gpu] Could not query max GPU clock — skipping clock lock.");
            return Self::none();
        };
        let ok = Command::new("nvidia-smi")
            .args(["-i", &index.to_string(), "-lgc", &max_clock.to_string()])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if ok {
            eprintln!("[gpu] Locked clocks to {max_clock} MHz for sustained inference throughput.");
            Self { index: Some(index) }
        } else {
            eprintln!("[gpu] Could not lock GPU clocks (try running as Administrator) — \
                       using driver default power management.");
            Self::none()
        }
    }
}

impl Drop for ClockLock {
    fn drop(&mut self) {
        if let Some(index) = self.index {
            let _ = Command::new("nvidia-smi")
                .args(["-i", &index.to_string(), "-rgc"])
                .output();
        }
    }
}

fn find_smi_index(device_name: &str) -> Option<u32> {
    let out = Command::new("nvidia-smi").arg("-L").output().ok()?;
    if !out.status.success() { return None; }
    let text = String::from_utf8_lossy(&out.stdout);
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("GPU ") else { continue };
        let Some((idx_str, tail)) = rest.split_once(':') else { continue };
        if tail.contains(device_name) || device_name.contains(tail.trim().split(" (UUID").next().unwrap_or("")) {
            return idx_str.trim().parse().ok();
        }
    }
    text.lines().next().and_then(|l| l.strip_prefix("GPU ")?.split_once(':')?.0.trim().parse().ok())
}

fn query_max_clock(index: u32) -> Option<u32> {
    let out = Command::new("nvidia-smi")
        .args(["-i", &index.to_string(), "--query-gpu=clocks.max.sm", "--format=csv,noheader,nounits"])
        .output().ok()?;
    if !out.status.success() { return None; }
    String::from_utf8_lossy(&out.stdout).trim().parse().ok()
}
