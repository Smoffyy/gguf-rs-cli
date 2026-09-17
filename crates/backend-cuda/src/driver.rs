//! CUDA Driver API, opened at run time.
//!
//! Nothing here links against CUDA. The driver library ships with the display driver, so
//! `nvcuda.dll` / `libcuda.so.1` is present on any machine with an NVIDIA GPU and absent
//! everywhere else - which is exactly the condition under which this backend should turn
//! itself off. Loading it dynamically means one binary runs on a machine with a GPU, a
//! machine without one, and a machine with no CUDA installed at all.
//!
//! The kernels are pre-compiled to PTX at build time and handed to the driver as a module;
//! the driver JITs them to the exact architecture present. That is what lets one build
//! target every GPU from Turing onward without a fat binary.

#![allow(non_camel_case_types, non_snake_case)]

use std::ffi::{c_void, CStr, CString};
use std::os::raw::{c_char, c_int, c_uint};

use gguf_core::{backend_err, Error, Result};
use libloading::{Library, Symbol};

pub type CUdevice = c_int;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUdeviceptr = u64;
pub type CUresult = c_int;

pub const CUDA_SUCCESS: CUresult = 0;

// Device attributes we actually branch on.
pub const ATTR_MAX_THREADS_PER_BLOCK: c_int = 1;
pub const ATTR_MAX_SHARED_PER_BLOCK: c_int = 8;
pub const ATTR_WARP_SIZE: c_int = 10;
pub const ATTR_MULTIPROCESSOR_COUNT: c_int = 16;
pub const ATTR_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
pub const ATTR_COMPUTE_CAPABILITY_MINOR: c_int = 76;

macro_rules! cuda_api {
    ($( fn $name:ident ( $($arg:ty),* $(,)? ) -> CUresult; )*) => {
        pub struct Driver {
            _lib: Library,
            $( pub $name: unsafe extern "C" fn($($arg),*) -> CUresult, )*
            get_error_string: Option<unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult>,
        }

        impl Driver {
            unsafe fn bind(lib: Library) -> std::result::Result<Self, String> {
                $(
                    let $name: Symbol<unsafe extern "C" fn($($arg),*) -> CUresult> =
                        lib.get(concat!(stringify!($name), "\0").as_bytes())
                            .map_err(|e| format!("{}: {e}", stringify!($name)))?;
                    let $name = *$name;
                )*
                let get_error_string = lib
                    .get::<unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult>(
                        b"cuGetErrorString\0",
                    )
                    .ok()
                    .map(|s| *s);
                Ok(Self { $( $name, )* get_error_string, _lib: lib })
            }
        }
    };
}

cuda_api! {
    fn cuInit(c_uint) -> CUresult;
    fn cuDeviceGetCount(*mut c_int) -> CUresult;
    fn cuDeviceGet(*mut CUdevice, c_int) -> CUresult;
    fn cuDeviceGetName(*mut c_char, c_int, CUdevice) -> CUresult;
    fn cuDeviceGetAttribute(*mut c_int, c_int, CUdevice) -> CUresult;
    fn cuDeviceTotalMem_v2(*mut usize, CUdevice) -> CUresult;
    fn cuCtxCreate_v2(*mut CUcontext, c_uint, CUdevice) -> CUresult;
    fn cuCtxDestroy_v2(CUcontext) -> CUresult;
    fn cuCtxSetCurrent(CUcontext) -> CUresult;
    fn cuMemGetInfo_v2(*mut usize, *mut usize) -> CUresult;
    fn cuModuleLoadData(*mut CUmodule, *const c_void) -> CUresult;
    fn cuModuleUnload(CUmodule) -> CUresult;
    fn cuModuleGetFunction(*mut CUfunction, CUmodule, *const c_char) -> CUresult;
    fn cuMemAlloc_v2(*mut CUdeviceptr, usize) -> CUresult;
    fn cuMemFree_v2(CUdeviceptr) -> CUresult;
    fn cuMemcpyHtoD_v2(CUdeviceptr, *const c_void, usize) -> CUresult;
    fn cuMemcpyDtoH_v2(*mut c_void, CUdeviceptr, usize) -> CUresult;
    fn cuMemcpyHtoDAsync_v2(CUdeviceptr, *const c_void, usize, CUstream) -> CUresult;
    fn cuMemcpyDtoDAsync_v2(CUdeviceptr, CUdeviceptr, usize, CUstream) -> CUresult;
    fn cuMemsetD8_v2(CUdeviceptr, u8, usize) -> CUresult;
    fn cuStreamCreate(*mut CUstream, c_uint) -> CUresult;
    fn cuStreamDestroy_v2(CUstream) -> CUresult;
    fn cuStreamSynchronize(CUstream) -> CUresult;
    fn cuLaunchKernel(
        CUfunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint,
        c_uint, CUstream, *mut *mut c_void, *mut *mut c_void,
    ) -> CUresult;
}

impl Driver {
    /// Open the driver library. Absence is an ordinary outcome, not a failure.
    pub fn open() -> Result<Self> {
        let candidates: &[&str] = if cfg!(windows) {
            &["nvcuda.dll"]
        } else if cfg!(target_os = "macos") {
            // Apple has not shipped a CUDA driver since 10.13; this exists so the error
            // names the real reason rather than "library not found".
            &["libcuda.dylib"]
        } else {
            &["libcuda.so.1", "libcuda.so"]
        };

        let mut last = String::new();
        for name in candidates {
            match unsafe { Library::new(name) } {
                Ok(lib) => {
                    return unsafe { Self::bind(lib) }
                        .map_err(|e| backend_err("cuda", format!("{name} is missing {e}")))
                }
                Err(e) => last = format!("{name}: {e}"),
            }
        }
        Err(Error::NoDevice(format!(
            "no NVIDIA driver library found ({last}); this is expected on a machine without \
             an NVIDIA GPU"
        )))
    }

    pub fn check(&self, code: CUresult, what: &str) -> Result<()> {
        if code == CUDA_SUCCESS {
            return Ok(());
        }
        let detail = match self.get_error_string {
            Some(f) => {
                let mut p: *const c_char = std::ptr::null();
                unsafe {
                    if f(code, &mut p) == CUDA_SUCCESS && !p.is_null() {
                        CStr::from_ptr(p).to_string_lossy().into_owned()
                    } else {
                        format!("error {code}")
                    }
                }
            }
            None => format!("error {code}"),
        };
        Err(backend_err("cuda", format!("{what}: {detail}")))
    }

    pub fn device_count(&self) -> Result<u32> {
        let mut n = 0;
        self.check(unsafe { (self.cuInit)(0) }, "cuInit")?;
        self.check(unsafe { (self.cuDeviceGetCount)(&mut n) }, "cuDeviceGetCount")?;
        Ok(n.max(0) as u32)
    }

    pub fn device_name(&self, dev: CUdevice) -> Result<String> {
        let mut buf = [0i8; 256];
        self.check(
            unsafe { (self.cuDeviceGetName)(buf.as_mut_ptr() as *mut c_char, 256, dev) },
            "cuDeviceGetName",
        )?;
        Ok(unsafe { CStr::from_ptr(buf.as_ptr() as *const c_char) }
            .to_string_lossy()
            .into_owned())
    }

    pub fn attribute(&self, dev: CUdevice, attr: c_int) -> Result<i32> {
        let mut v = 0;
        self.check(
            unsafe { (self.cuDeviceGetAttribute)(&mut v, attr, dev) },
            "cuDeviceGetAttribute",
        )?;
        Ok(v)
    }

    pub fn total_memory(&self, dev: CUdevice) -> Result<u64> {
        let mut bytes = 0usize;
        self.check(unsafe { (self.cuDeviceTotalMem_v2)(&mut bytes, dev) }, "cuDeviceTotalMem")?;
        Ok(bytes as u64)
    }

    pub fn function(&self, module: CUmodule, name: &str) -> Result<CUfunction> {
        let c = CString::new(name).unwrap();
        let mut f: CUfunction = std::ptr::null_mut();
        self.check(
            unsafe { (self.cuModuleGetFunction)(&mut f, module, c.as_ptr()) },
            &format!("looking up kernel {name}"),
        )?;
        Ok(f)
    }
}

/// A kernel launch, built argument by argument.
///
/// The driver takes an array of pointers to arguments, so every value has to outlive the
/// launch. Collecting them here keeps that lifetime obvious at the call site.
pub struct Launch {
    storage: Vec<[u8; 8]>,
    sizes: Vec<usize>,
}

impl Launch {
    pub fn new() -> Self {
        Self { storage: Vec::with_capacity(12), sizes: Vec::with_capacity(12) }
    }

    fn push_bytes(&mut self, bytes: &[u8]) {
        let mut slot = [0u8; 8];
        slot[..bytes.len()].copy_from_slice(bytes);
        self.storage.push(slot);
        self.sizes.push(bytes.len());
    }

    pub fn ptr(mut self, p: CUdeviceptr) -> Self {
        self.push_bytes(&p.to_ne_bytes());
        self
    }

    pub fn i32(mut self, v: i32) -> Self {
        self.push_bytes(&v.to_ne_bytes());
        self
    }

    pub fn u64(mut self, v: u64) -> Self {
        self.push_bytes(&v.to_ne_bytes());
        self
    }

    pub fn f32(mut self, v: f32) -> Self {
        self.push_bytes(&v.to_ne_bytes());
        self
    }

    /// SAFETY: the caller guarantees the kernel's signature matches the pushed arguments.
    pub unsafe fn run(
        &mut self,
        drv: &Driver,
        func: CUfunction,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared: u32,
        stream: CUstream,
        what: &str,
    ) -> Result<()> {
        let mut params: Vec<*mut c_void> = self
            .storage
            .iter_mut()
            .map(|slot| slot.as_mut_ptr() as *mut c_void)
            .collect();
        drv.check(
            (drv.cuLaunchKernel)(
                func,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                shared,
                stream,
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            ),
            what,
        )
    }
}

impl Default for Launch {
    fn default() -> Self {
        Self::new()
    }
}
