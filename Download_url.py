from urllib.parse import urlparse
import os
import time
from urllib.request import urlopen
import urllib
import whois
from Cython.Tempita._tempita import html
from ipwhois import IPWhois
from pprint import pprint
import socket
import dns.resolver
import requests
import pickle
import datetime
from bs4 import BeautifulSoup
import sys
import re
import logging
import traceback
import tldextract

logger = logging.getLogger('root')
whois_info = {}
def dns_lookup(domain):
    ids = ['NONE',
        'A',
        'NS',
        'CNAME',
        'PTR',
        'MX',
        'SRV',
        'IXFR',
        'AXFR',
        'HINFO',
        'TLSA',
        'URI'
    ]
    lists=[]
    for a in ids:
        try:
            answers = dns.resolver.query(domain, a)
            for rdata in answers:
                val=a + ' : '+ rdata.to_text()
                lists.append(val)

        except Exception as e:
            pass
    return lists

def download_url(rawurl):
    html, dns_lookup_output, IPs, ipwhois, whois_output, content, domain, html_time, dns_lookup_time, ipwhois_time, Error = 0,0,0,0,0,0,0,0,0,0,0
    headers = requests.utils.default_headers()

    headers.update(
        {
            'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:62.0) Gecko/20100101 Firefox/62.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q = 0.8',
            'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q = 0.7',
            'Keep-Alive': '300',
            'Pragma': 'no-cache',
            'Cache-Control': 'no-cache',
            'Accept-Language': '*',
            'Accept-Encoding': 'gzip, deflate'
        }
    )

    url = rawurl.strip().rstrip('\n')
    if url == '':
        pass
    try:
        t0 = time.time()
        html = requests.get(url=url, headers = headers, timeout = 20)
        if html.status_code != 200:
        	pass
        else:
            parsed = BeautifulSoup(html.text, 'html.parser')
            language = parsed.find("html").get('lang')
            if language != None and language != 'en':
                pass
        html_time = time.time() - t0
        content = html.text
        landing_url = html.url
    except Exception as e:
        print("Exception: HTML Error :{}".format(e))
        print("html, content=''")
        html=''
        content=''
        html_time=''

    try:           
        extracted = tldextract.extract(landing_url)
        domain = "{}.{}".format(extracted.domain, extracted.suffix)

        t0 = time.time()
        dns_lookup_output=dns_lookup(domain)
        dns_lookup_time = time.time() - t0

        try:
            IPs = list(map(lambda x: x[4][0], socket.getaddrinfo(domain, 80, type=socket.SOCK_STREAM)))
        except socket.gaierror:
            IPs = list(map(lambda x: x[4][0], socket.getaddrinfo("www." + domain, 80, type=socket.SOCK_STREAM)))

        t0 = time.time()
        for ip in IPs:
            obj = IPWhois(ip)
            ipwhois = obj.lookup_whois(get_referral=True)
        ipwhois_time = time.time() - t0

    except Exception as e:
        logger.error("Exception: Domain Error: {}".format(e))
        logger.error("domain, dns_lookup_output, dns_lookup_time, IPs, ipwhois, ipwhois_time =''")
        domain=''
        dns_lookup_output=''
        dns_lookup_time=''
        IPs=''
        ipwhois=''
        ipwhois_time=''

        
    try:
        if domain in whois_info:
            whois_output = whois_info[domain]
        else:
            whois_output = whois.whois(domain)
            whois_info[domain] = whois_output
        time.sleep(3)
    except Exception as e:
        logger.error("Exception: Domain Error: {}".format(e))
        logger.error("domain, dns_lookup_output, dns_lookup_time, IPs, ipwhois, ipwhois_time =''")
        whois_output=''


        

    # except Exception as e:
    #     logger.error(e)
    #     logger.error(traceback.format_exc())
    #     Error=1
    return html,dns_lookup_output, IPs, ipwhois, whois_output, content, domain, html_time, dns_lookup_time, ipwhois_time

def visible(element):
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return False
    elif re.match('<!--.*-->', str(element.encode('utf-8'))):
        return False
    return True

def download_url_content(rawurl):
    html = urllib.request.urlopen('http://bgr.com/2014/10/15/google-android-5-0-lollipop-release/')
    soup = BeautifulSoup(html, "html5lib")
    data = soup.findAll(text=True)
    result = filter(visible, data)
    return list(result)

if __name__ == "__main__":
    data = download_url_content("http://docs.python-requests.org/en/master/user/quickstart/")
    logging.info(data)
