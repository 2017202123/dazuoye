import nltk
from xpinyin import Pinyin

punct = set(u''':!),.:;?.]}¢'"、。〉》」』〕〗〞︰︱︳﹐､﹒
﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､￠
々‖•·ˇˉ―′’”([{£¥'"‵〈《「『〔〖（［｛￡￥〝︵︷︹︻
︽︿﹁﹃﹙﹛﹝（｛“‘_…/''')

stemmer = nltk.stem.PorterStemmer()


def stem(word):
    return stemmer.stem(word)


def clean_sentence(text, stemming=False):
    for token in punct:
        text = text.replace(token, "")
    words = text.split()
    if stemming:
        stemmed_words = []
        for w in words:
            stemmed_words.append(stem(w))
        words = stemmed_words
    return " ".join(words)


def clean_name(name):
    if name is None or name =='':
        return ""
    Chinese=True
    for _char in name:
        # if '\u4e00' <= _char <= '\u9fa5': # 处理中文字符
        if (_char<'\u4e00' or _char>'\u9fa5') and _char!=' ': # 如果是英文字符
            Chinese = False
            break
    if Chinese:
        p = Pinyin()
        new_name = p.get_pinyin(name, ',')
        if len(name) == 3:  # 中文三个字的名字
            # new_name[new_name.rfind('-')] = ','
            new_name = new_name.replace(',', '-', 1)
            new_name = new_name.replace(',', '', 1)
            x = [k.strip() for k in new_name.lower().strip().replace(".", " ").replace("-", " ").split()]
            # 交换名和姓氏
            tmp = x[len(x) - 1]
            x[len(x) - 1] = x[0]
            x[0] = tmp
        elif ' ' in name:  # 四个字的名字（日文名翻译来的）
            new_name = new_name.replace(', ,', '-')
            new_name = new_name.replace(',', '')
            x = [k.strip() for k in new_name.lower().strip().replace(".", " ").replace("-", " ").split()]
        else:  # 两个字的名字
            new_name = new_name.replace(',', '-')
            x = [k.strip() for k in new_name.lower().strip().replace(".", " ").replace("-", " ").split()]
            # 交换名和姓氏
            tmp = x[len(x) - 1]
            x[len(x) - 1] = x[0]
            x[0] = tmp
        # ret = '_'.join([x[1], x[0]])
        ret = "_".join(x)
        # print(name, new_name,ret)
    else:
        x = [k.strip() for k in name.lower().strip().replace(".", " ").replace("-", " ").split()]
        ret = "_".join(x)
    return ret


def clean_name_author(name):
    if name is None or name == '':
        return ""
    x = [k.strip() for k in name.lower().strip().replace(".", " ").replace("_", " ").split()]
    tmp = x[len(x)-1]
    x[len(x) - 1] = x[0]
    x[0] = tmp
    return "_".join(x)
