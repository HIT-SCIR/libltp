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

/// Error type centralizing all possible errors
#[non_exhaustive]
#[derive(Error, Debug)]
enum PyO3LTPError {
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
}

#[pymodule]
fn pyltp(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<LTP>()?;
    Ok(())
}