from pyltp import LTP


def main():
    ltp = LTP("path/to/model")

    sentences = ["他叫汤姆去拿外衣！", "我爱赛尔！"]
    result = ltp.pipeline(sentences)
    print(result)


if __name__ == '__main__':
    main()
