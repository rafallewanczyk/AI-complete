import os
import re
import numpy as np
from tqdm import tqdm
from selenium import webdriver
from time import sleep
import requests
import lxml.html
import unidecode


def clean_content(x):
    x = unidecode.unidecode(x)
    x = re.sub('#.*$', '', x, flags=re.MULTILINE)
    x = re.sub("'''[\s\S]*?'''", '', x, flags=re.MULTILINE)
    x = re.sub('"""[\s\S]*?"""', '', x, flags=re.MULTILINE)
    x = re.sub('^[\t]+\n', '', x, flags=re.MULTILINE)
    x = re.sub('^[ ]+\n', '', x, flags=re.MULTILINE)
    x = re.sub('\n[\n]+', '\n\n', x, flags=re.MULTILINE)

    x += '\nEOF\n'
    return x


username = 'rafallewanczyk'
password = 'guer1llA'

search_query = 'import keras'

minsize = 30000

outdir = r'.\\pysource\\'

driver = webdriver.Chrome(r'.\\chromedriver.exe')
driver.get('https://github.com/login')
driver.find_element_by_id('login_field').send_keys(username)
driver.find_element_by_id('password').send_keys(password)
driver.find_element_by_id('password').submit()

query = search_query + ' filename:*.py size:>' + str(minsize) + ' language:Python'

links = set()
counts = [0, 0, 0]
counter = 1
data = ''

pbar = tqdm(range(1, 101))

for i in pbar:
    try:
        sleep(np.random.uniform(2))

        url = "https://github.com/search?p={}&q={}&ref=searchresults&type=Code&utf8=%E2%9C%93"
        driver.get(url.format(i, query))

        tree = lxml.html.fromstring(driver.page_source)
        page_links = list(set([x for x in tree.xpath('//a/@href') if "/blob/" in x and "#" not in x]))
        pbar.set_description("Page #{}. Found {} links.".format(i, len(page_links)))

        counts.append(len(page_links))
        k = sum(counts[-3:])
        if k == 0:
            break

        for link in page_links:
            if link not in links:
                try:
                    sleep(np.random.uniform(2))
                    url = "https://github.com" + link
                    html = requests.get(url).text
                    tree = lxml.html.fromstring(html)
                    xpath = '//*[@class="blob-code blob-code-inner js-file-line"]'
                    content = "\n".join([x.text_content() for x in tree.xpath(xpath)])
                    content = clean_content(content)
                    with open(outdir + '\\' + f'{counter}.py', 'w') as f:
                        f.write(content)
                    pbar.set_description(
                        "Page #{}. File #{}. Total data size{:.3f}MB".format(i, counter, len(data) / 1e6))
                    counter += 1
                except:
                    pass
        links.update(page_links)
    except KeyboardInterrupt:
        break

print("no more results found")
