__author__ = 'ffuuugor'
from os import listdir
import os
import re
from bs4 import BeautifulSoup
# from es_index import index_docs

def process_dir(book_dir, func):
    htmls = filter(lambda x: re.match("[0-9]+.html", x), listdir(book_dir))
    # htmls = ["1.html"]
    docs = []
    for filename in htmls:
        with open(os.path.join(book_dir,filename)) as f:
            html_doc = f.read()
            parser = BeautifulSoup(html_doc, "html.parser")
            docs.extend(func(parser))

    return docs

def one_per_html_docs(book_dir):
    def apply(parser):
        return [re.sub("\s+"," ",parser.get_text())]

    return process_dir(book_dir, apply)

def one_per_h1_docs(book_dir):
    def apply(parser):
        docs = []
        currdoc = None
        for elem in parser.body.children:
            if elem.name == "h1":
                if currdoc is not None:
                    docs.append(currdoc)

                currdoc = elem.string.strip()
            elif elem.name is not None:
                if currdoc is not None and elem.string is not None and len(elem.string.strip()) > 0:
                    currdoc += ". " + elem.string.strip()
        return docs

    return process_dir(book_dir, apply)


if __name__ == '__main__':
    docs = one_per_h1_docs("data/ck12_book/OEBPS")

    print len(filter(lambda x: len(x.strip()) > 0, docs))
    # print "huge"
    # index_docs(one_per_html_docs("OEBPS"), index="ck12_snowball")
    # print "h1"
    # index_docs(one_per_h1_docs("OEBPS"), index="ck12_h1_snowball")

