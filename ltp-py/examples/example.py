import pyarrow as pa
from pyltp import LTP


def main():
    ltp = LTP("../../onnx-small")

    sentences = ["他叫汤姆去拿外衣！", "我爱赛尔！"]
    result = ltp.pipeline(sentences)
    print(result)

    sentences = pa.array(sentences, type=pa.string())
    seg, pos, ner, srl, dep, sdp = ltp.pipeline_arrow(sentences)

    record_batch = pa.record_batch([seg, pos, ner, srl, dep, sdp], names=['seg', 'pos', 'ner', 'srl', 'dep', 'sdp'])
    pystruct = record_batch.to_pydict()
    print(pystruct)


if __name__ == '__main__':
    main()
