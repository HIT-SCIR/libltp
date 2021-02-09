use std::error;
use std::fmt;
use std::sync::{Arc, Mutex};

use std::ffi::CStr;
use std::os::raw::c_char;
use std::os::raw::c_void;

use ltp_rs::{LTP as Interface, LTPError};
use ltp_rs::preinclude::arrow::array::{make_array_from_raw, ArrayRef};
use ltp_rs::preinclude::arrow::error::ArrowError;
use ltp_rs::preinclude::arrow::ffi;
use std::ptr::slice_from_raw_parts_mut;


#[repr(C)]
pub struct LTPResultArray {
    arrays: *mut *const ffi::FFI_ArrowArray,
    schemes: *mut *const ffi::FFI_ArrowSchema,
}

#[no_mangle]
pub extern fn ltp_init(path: *const c_char) -> *mut c_void {
    let path = unsafe {
        assert!(!path.is_null());
        CStr::from_ptr(path)
    };
    let path = path.to_str().unwrap();
    let interface = Box::new(Interface::new(path).unwrap());
    Box::into_raw(interface) as *mut c_void
}

#[no_mangle]
pub extern fn ltp_release(interface: *mut c_void) {
    unsafe {
        assert!(!interface.is_null());
        Box::from_raw(interface as *mut Interface);
    }
}

#[no_mangle]
pub extern fn ltp_pipeline(
    interface: *mut c_void,
    array_ptr: *const ffi::FFI_ArrowArray,
    schema_ptr: *const ffi::FFI_ArrowSchema,
    result_ptr: *mut LTPResultArray,
) {
    let mut interface = unsafe {
        assert!(!interface.is_null());
        Box::from_raw(interface as *mut Interface)
    };

    let array = unsafe {
        make_array_from_raw(array_ptr, schema_ptr)
    }.unwrap();

    let result = interface.pipeline_arrow(array).unwrap();

    let result_array_slice = unsafe {
        slice_from_raw_parts_mut((*result_ptr).arrays, result.len())
    };

    let result_scheme_slice = unsafe {
        slice_from_raw_parts_mut((*result_ptr).schemes, result.len())
    };

    let mut iter = 0;
    for array in result {
        let (array_pointer, schema_pointer) = array.to_raw().unwrap();
        unsafe {
            (*result_array_slice)[iter] = array_pointer;
            (*result_scheme_slice)[iter] = schema_pointer;
        }
        iter += 1;
    }

    Box::into_raw(interface);
}

#[cfg(test)]
mod tests {
    use crate::ltp_init;


    #[test]
    fn test_interface() {}
}
