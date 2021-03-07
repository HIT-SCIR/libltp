use jni::{
    errors::Error as JNIError,
    objects::{JClass, JList, JObject, JString},
    sys::{jlong, jobject},
    JNIEnv,
};

use jni::sys::jint;
use ltp_rs::{
    preinclude::thiserror::{self, Error},
    LTPError, LTP as Interface,
};

/// Error type centralizing all possible errors
#[non_exhaustive]
#[derive(Error, Debug)]
enum JniLTPError {
    #[error("{0}")]
    LTPError(LTPError),
    #[error("{0}")]
    JNIError(JNIError),
}

type Result<T> = std::result::Result<T, JniLTPError>;

impl From<LTPError> for JniLTPError {
    fn from(err: LTPError) -> JniLTPError {
        JniLTPError::LTPError(err)
    }
}

impl From<JNIError> for JniLTPError {
    fn from(err: JNIError) -> JniLTPError {
        JniLTPError::JNIError(err)
    }
}

fn new_java_list<'a, 'b>(env: &'a JNIEnv) -> Result<JList<'a, 'b>>
where
    'a: 'b,
{
    let jclass = env.find_class("java/util/ArrayList")?;
    let jobject = env.new_object(jclass, "()V", &[])?;
    let result = JList::from_env(&env, jobject)?;
    Ok(result)
}

fn ltp_rust_init(
    env: JNIEnv,
    _class: JClass,
    path: JString,
    num_threads: jint,
    device_id: jint,
) -> Result<jlong> {
    let path: String = env.get_string(path)?.into();

    #[cfg(feature = "cuda")]
    let interface = if device_id >= 0 {
        Box::new(Interface::new_with_cuda(&path, num_threads as i16, n)?)
    } else {
        Box::new(Interface::new(&path, num_threads as i16)?)
    };
    #[cfg(not(feature = "cuda"))]
    let interface = Box::new(Interface::new(&path, num_threads as i16)?);
    Ok(Box::into_raw(interface) as jlong)
}

fn ltp_rust_pipeline(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    sentences: JObject,
) -> Result<jobject> {
    let sentences = JList::from_env(&env, sentences)?;
    let length = sentences.size()?.into();
    let mut batch_sentences = Vec::new();
    for idx in 0..length {
        let sentence: JObject = sentences.get(idx)?.unwrap();
        let sentence = JString::from(sentence);
        let sentence: String = env.get_string(sentence)?.into();
        batch_sentences.push(sentence);
    }

    let mut interface = unsafe {
        let ptr = ptr as *mut Interface;
        assert!(!ptr.is_null());
        Box::from_raw(ptr)
    };

    let results = interface.pipeline_batch(&batch_sentences)?;
    Box::into_raw(interface);

    let java_list = new_java_list(&env)?;
    for result in results {
        let java_one_list = new_java_list(&env)?;

        let seg = new_java_list(&env)?;
        if result.seg.is_some() {
            for word in result.seg.unwrap() {
                seg.add(env.new_string(word)?.into())?;
            }
        }
        java_one_list.add(JObject::from(seg))?;

        let pos = new_java_list(&env)?;
        if result.pos.is_some() {
            for tag in result.pos.unwrap() {
                pos.add(env.new_string(tag)?.into())?;
            }
        }
        java_one_list.add(JObject::from(pos))?;

        let ner = new_java_list(&env)?;
        if result.ner.is_some() {
            for tag in result.ner.unwrap() {
                ner.add(env.new_string(tag)?.into())?;
            }
        }
        java_one_list.add(JObject::from(ner))?;

        let srl = new_java_list(&env)?;
        if result.srl.is_some() {
            for tags in result.srl.unwrap() {
                let srl_tags = new_java_list(&env)?;
                for tag in tags {
                    srl_tags.add(env.new_string(tag)?.into())?;
                }
                srl.add(JObject::from(srl_tags))?;
            }
        }
        java_one_list.add(JObject::from(srl))?;

        let dep = new_java_list(&env)?;
        if result.dep.is_some() {
            for tag in result.dep.unwrap() {
                dep.add(env.new_string(format!("{}:{}", tag.arc, tag.rel))?.into())?;
            }
        }
        java_one_list.add(JObject::from(dep))?;

        let sdp = new_java_list(&env)?;
        if result.sdp.is_some() {
            for tag in result.sdp.unwrap() {
                sdp.add(
                    env.new_string(format!("{}:{}:{}", tag.src, tag.tgt, tag.rel))?
                        .into(),
                )?;
            }
        }
        java_one_list.add(JObject::from(sdp))?;

        java_list.add(JObject::from(java_one_list))?;
    }

    Ok(java_list.into_inner())
}

#[no_mangle]
pub extern "system" fn Java_cn_edu_hit_ir_LTP_rust_1init(
    env: JNIEnv,
    _class: JClass,
    path: JString,
    num_threads: jint,
    device_id: jint,
) -> jlong {
    let result = ltp_rust_init(env, _class, path, num_threads, device_id);

    match result {
        Ok(res) => res,
        Err(err) => {
            env.throw_new("java/lang/NullPointerException", format!("{}", err))
                .expect("jni native error!");
            0
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_cn_edu_hit_ir_LTP_rust_1release(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
) {
    unsafe {
        let ptr = ptr as *mut Interface;
        if ptr.is_null() {
            env.throw_new(
                "java/lang/NullPointerException",
                "release null native ltp pointer",
            )
            .expect("jni native error!");
        } else {
            Box::from_raw(ptr);
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_cn_edu_hit_ir_LTP_rust_1pipeline(
    env: JNIEnv,
    _class: JClass,
    ptr: jlong,
    sentences: JObject,
) -> jobject {
    let result = ltp_rust_pipeline(env, _class, ptr, sentences);
    match result {
        Ok(res) => res,
        Err(err) => {
            env.throw_new("java/lang/Exception", format!("{}", err))
                .expect("jni native error!");
            std::ptr::null_mut()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Java_cn_edu_hit_ir_LTP_rust_1init, Java_cn_edu_hit_ir_LTP_rust_1pipeline,
        Java_cn_edu_hit_ir_LTP_rust_1release,
    };
    use jni::objects::{JList, JObject};
    use jni::{errors::Result as JNIResult, InitArgsBuilder, JNIVersion, JavaVM};

    #[test]
    fn test_display() -> JNIResult<()> {
        // Build the VM properties
        let jvm_args = InitArgsBuilder::new()
            // Pass the JNI API version (default is 8)
            .version(JNIVersion::V8)
            // You can additionally pass any JVM options (standard, like a system property,
            // or VM-specific).
            // Here we enable some extra JNI checks useful during development
            .option("-Xcheck:jni")
            .build()
            .unwrap();

        // Create a new VM
        let jvm = JavaVM::new(jvm_args)?;

        // Attach the current thread to call into Java — see extra options in
        // "Attaching Native Threads" section.
        //
        // This method returns the guard that will detach the current thread when dropped,
        // also freeing any local references created in it
        let env = jvm.attach_current_thread_permanently()?;

        let jclass = env.find_class("java/util/ArrayList")?;
        let jobject = env.new_object(jclass, "()V", &[])?;
        let jlist = JList::from_env(&env, jobject)?;
        let jstring = env.new_string("他叫汤姆去拿外衣")?;
        jlist
            .add(JObject::from(jstring))
            .expect("list add object error");

        let jstring = env.new_string("models/small")?;
        let ptr = Java_cn_edu_hit_ir_LTP_rust_1init(env, jclass, jstring, 16, 0);
        Java_cn_edu_hit_ir_LTP_rust_1pipeline(env, jclass, ptr, jobject);
        Java_cn_edu_hit_ir_LTP_rust_1release(env, jclass, ptr);

        println!("{:?}", jobject);

        Ok(())
    }
}
