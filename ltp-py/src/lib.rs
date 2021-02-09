use std::sync::{Arc, Mutex};

use pyo3::{
    prelude::*,
    exceptions::PyOSError,
    types::{PyUnicode, PyList, PyDict},
};

use ltp_rs::{
    LTP as Interface, LTPError,
    preinclude::{thiserror::{self, Error}, itertools::Itertools},
};

#[cfg(feature = "arrow")]
use ltp_rs::preinclude::arrow::{
    ffi, error::ArrowError,
    array::{make_array_from_raw, ArrayRef},
};


/// Error type centralizing all possible errors
#[non_exhaustive]
#[derive(Error, Debug)]
enum PyO3LTPError {
    #[cfg(feature = "arrow")]
    #[error("{0}")]
    ArrowError(ArrowError),
    #[error("{0}")]
    LTPError(LTPError),
}

#[pyclass]
struct LTP {
    interface: Arc<Mutex<Interface>>,
}

unsafe impl Send for LTP {}

unsafe impl Sync for LTP {}

impl From<LTPError> for PyO3LTPError {
    fn from(err: LTPError) -> PyO3LTPError {
        PyO3LTPError::LTPError(err)
    }
}

#[cfg(feature = "arrow")]
impl From<ArrowError> for PyO3LTPError {
    fn from(err: ArrowError) -> PyO3LTPError {
        PyO3LTPError::ArrowError(err)
    }
}

impl From<PyO3LTPError> for PyErr {
    fn from(err: PyO3LTPError) -> PyErr {
        PyOSError::new_err(err.to_string())
    }
}

#[pymethods]
impl LTP {
    #[new]
    fn new(path: &PyUnicode) -> PyResult<Self> {
        let interface = Interface::new(&path.to_string())
            .map_err(|e| PyO3LTPError::from(LTPError::from(e)))?;
        Ok(LTP { interface: Arc::new(Mutex::new(interface)) })
    }

    fn pipeline(&self, py: Python, ob: PyObject) -> PyResult<PyObject> {
        let ob: &PyList = ob.cast_as(py)?;
        let array = ob.iter().map(|x| x.to_string()).collect_vec();

        let mut interface_guard = self.interface
            .lock()
            .expect("Failed to acquire lock: another thread panicked?");

        let results = interface_guard.pipeline_batch(&array)
            .map_err(|e| PyO3LTPError::from(e))?;

        let list = PyList::empty(py);

        for result in results {
            let one_sentence = PyDict::new(py);

            let seg = PyList::empty(py);
            if result.seg.is_some() {
                for word in result.seg.unwrap() {
                    seg.append(word)?;
                }
            }
            one_sentence.set_item("seg", seg)?;

            let pos = PyList::empty(py);
            if result.pos.is_some() {
                for tag in result.pos.unwrap() {
                    pos.append(tag)?;
                }
            }
            one_sentence.set_item("pos", pos)?;

            let ner = PyList::empty(py);
            if result.ner.is_some() {
                for tag in result.ner.unwrap() {
                    ner.append(tag)?;
                }
            }
            one_sentence.set_item("ner", ner)?;

            let srl = PyList::empty(py);
            if result.srl.is_some() {
                for tags in result.srl.unwrap() {
                    let srl_tags = PyList::empty(py);
                    for tag in tags {
                        srl_tags.append(tag)?;
                    }
                    srl.append(srl_tags)?;
                }
            }
            one_sentence.set_item("srl", srl)?;

            let dep = PyList::empty(py);
            if result.dep.is_some() {
                for tag in result.dep.unwrap() {
                    dep.append((tag.arc, tag.rel))?;
                }
            }
            one_sentence.set_item("dep", dep)?;

            let sdp = PyList::empty(py);
            if result.sdp.is_some() {
                for tag in result.sdp.unwrap() {
                    sdp.append((tag.src, tag.tgt, tag.rel))?;
                }
            }
            one_sentence.set_item("sdp", sdp)?;

            list.append(one_sentence)?;
        }
        Ok(list.to_object(py))
    }

    #[cfg(feature = "arrow")]
    fn pipeline_arrow(&self, py: Python, ob: PyObject) -> PyResult<PyObject> {
        let array = to_rust(py, ob)?;

        let mut interface_guard = self.interface
            .lock()
            .expect("Failed to acquire lock: another thread panicked?");

        let result = interface_guard.pipeline_arrow(array)
            .map_err(|e| PyO3LTPError::from(e))?;

        let list = PyList::empty(py);

        for array in result {
            list.append(to_py(py, array)?)?;
        }
        Ok(list.to_object(py))
    }
}

#[cfg(feature = "arrow")]
fn to_rust(py: Python, ob: PyObject) -> PyResult<ArrayRef> {
    // prepare a pointer to receive the Array struct
    let (array_pointer, schema_pointer) =
        ffi::ArrowArray::into_raw(unsafe { ffi::ArrowArray::empty() });

    // make the conversion through PyArrow's private API
    // this changes the pointer's memory and is thus unsafe. In particular, `_export_to_c` can go out of bounds
    ob.call_method1(
        py,
        "_export_to_c",
        (array_pointer as usize, schema_pointer as usize),
    )?;

    let array = unsafe { make_array_from_raw(array_pointer, schema_pointer) }
        .map_err(|e| PyO3LTPError::from(e))?;
    Ok(array)
}

#[cfg(feature = "arrow")]
fn to_py(py: Python, array: ArrayRef) -> PyResult<PyObject> {
    let (array_pointer, schema_pointer) =
        array.to_raw().map_err(|e| PyO3LTPError::from(e))?;

    let pa = py.import("pyarrow")?;

    let array = pa.getattr("Array")?.call_method1(
        "_import_from_c",
        (array_pointer as usize, schema_pointer as usize),
    )?;
    Ok(array.to_object(py))
}

#[pymodule]
fn pyltp(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<LTP>()?;
    Ok(())
}