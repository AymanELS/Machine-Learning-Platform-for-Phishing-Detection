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
    #ids = [
    #    'NONE',
    #    'A',
    #    'NS',
    #    'MD',
    #    'MF',
    #    'CNAME',
    ##    'SOA',
    ##    'MB',
    #    'MG',
    #    'MR',
    #    'NULL',
    #    'WKS',
    #    'PTR',
    #    'HINFO',
    #    'MINFO',
    #    'MX',
    #    'TXT',
    #    'RP',
    #    'AFSDB',
    #    'X25',
    #    'ISDN',
    #    'RT',
    #    'NSAP',
    #    'NSAP-PTR',
    ##    'SIG',
    #    'KEY',
    #    'PX',
    #    'GPOS',
    #    'AAAA',
    #    'LOC',
    #    'NXT',
    #    'SRV',
    #    'NAPTR',
    #    'KX',
    #    'CERT',
    #    'A6',
    #    'DNAME',
    #    'OPT',
    #    'APL',
    ####    'DS',
    ##    'SSHFP',
    #    'IPSECKEY',
    #    'RRSIG',
    #    'NSEC',
    ####    'DNSKEY',
    #    'DHCID',
    #    'NSEC3',
    #    'NSEC3PARAM',
    ###    'TLSA',
    #    'HIP',
    #    'CDS',
    #    'CDNSKEY',
    #    'CSYNC',
    #    'SPF',
    #    'UNSPEC',
    #    'EUI48',
    #    'EUI64',
    #    'TKEY',
    #    'TSIG',
    #    'IXFR',
    #    'AXFR',
    #    'MAILB',
    #    'MAILA',
    #    'ANY',
    #    'URI',
    #    'CAA',
    #    'TA',
    #    'DLV',
    #]
    lists=[]
    for a in ids:
        try:
            #print("entered in loop")
            answers = dns.resolver.query(domain, a)
            for rdata in answers:
                val=a + ' : '+ rdata.to_text()
                lists.append(val)

        except Exception as e:
            pass
    return lists

def download_url(rawurl):
    html, dns_lookup_output, IPs, ipwhois, whois_output, content, domain, html_time, dns_lookup_time, ipwhois_time, Error= 0,0,0,0,0,0,0,0,0,0,0
    headers = requests.utils.default_headers()

    headers.update(
        {
            'User-Agent': 'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.5) Gecko/20091102 Firefox/3.5.5 (.NET CLR 3.5.30729)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q = 0.8',
            'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q = 0.7',
            'Keep-Alive': '300',
            'Pragma': 'no-cache',
            'Cache-Control': 'no-cache',
            'Accept-Language': '*'
        }
    )
    #urls_file = open("url_list_URL_CHANGED.txt", 'w')
    #urls_file = open(file_list_urls, 'w')
    #counter = 19267   
    #for rawurl in open("file_folder_list_random_continue_filtered.txt", 'rU'):
    #for rawurl in open(file_output_urls, 'rU'):
    #    if counter == 400000:
    #        break
    #print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    #print(rawurl)
    try:
        url = rawurl.strip().rstrip('\n')
        if url == '':
            pass
        
        #parsed_url = urlparse(url)
        #domain = '{uri.netloc}'.format(uri=parsed_url)
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
        landing_url = html.url            
        parsed_url = urlparse(landing_url)
        domain = '{uri.netloc}'.format(uri=parsed_url)

            #html_output_file_name = ("URL_features/htmls/" + url.replace("/", "_"))[:100] + ".html"
            #header_output_file_name = ("URL_features/headers/" + url.replace("/", "_"))[:100] + ".txt"
            #ipwhois_output_file_name = ("URL_features/headers/" + url.replace("/", "_"))[:100] + ".ipwhois"
            ##whois_output_file_name = ("URL_features/headers/" + url.replace("/", "_"))[:100] + ".whois"
            #http_response_output_file_name = ("URL_features/headers/" + url.replace("/", "_"))[:100] + ".http_response"
            #dns_output_file_name = ("URL_features/headers/" + url.replace("/", "_"))[:100] + ".dns"
            #html_output_file = open(html_output_file_name, "w")
            #header_output_file = open(header_output_file_name, "w")
            #################### HTTP Response ################################

            #http_response_output_file = open(http_response_output_file_name, "wb")
            #pickle.dump(html, http_response_output_file)
            #if html.history:
                #print("Request was redirected", file=header_output_file)
                #for resp in html.history:
                #    print(resp.status_code, resp.url, file=header_output_file)
            #http_response_output_file.close()
            #print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++", file=header_output_file)
            #print("HTML Header", file=header_output_file)
            #print(html.headers, file=header_output_file)
            #print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++", file=header_output_file)

            #################### DNS and IP ################################

            #dns_output_file = open(dns_output_file_name, "w")
            #print("DNS Lookup", file=dns_output_file)
        t0 = time.time()
        #dns_lookup=dns_lookup(domain, output = dns_output_file)
        dns_lookup_output=dns_lookup(domain)
        #print("dns_lookup_output " + dns_lookup_output)
        dns_lookup_time = time.time() - t0
        #print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++", file=dns_output_file)
        #print("IP", file=dns_output_file)
        IPs = list(map(lambda x: x[4][0], socket.getaddrinfo(domain, 80, type=socket.SOCK_STREAM)))
        #print(','.join(IPs),
        #      file=dns_output_file)
        #dns_output_file.close()

            #################### IPwhois ################################
            #print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++", file=header_output_file)

            #ipwhois_output_file = open(ipwhois_output_file_name, "wb")
            #print("whois", file=header_output_file)
        t0 = time.time()
        for ip in IPs:
            obj = IPWhois(ip)
            ipwhois = obj.lookup_whois(get_referral=True)
            #pickle.dump(ipwhois, ipwhois_output_file)
            #pprint(results, header_output_file)
        ipwhois_time = time.time() - t0
        #ipwhois_output_file.close()


        #################### whois ################################            
        #whois_output_file = open(whois_output_file_name, "wb")
        #if domain in whois_info:
        #    pickle.dump(whois_info.get(domain), whois_output_file)
        #else:
        whois_output = whois.whois(domain)
        #pickle.dump(whois_output, whois_output_file)
        #whois_info[domain] = d

        #whois_output_file.close()
        #d = whois.whois(domain)
        #pickle.dump(d, whois_output_file)
        #whois_output_file.close()
        time.sleep(3)

        content = html.text

            #print(content, file=html_output_file)
            #print(content.decode(html.headers.get_content_charset()),  file=html_output_file)
            #html_output_file.close()
            #print(domain, file=header_output_file)
            #print(html_time, file=header_output_file)
            #print(dns_lookup_time, file=header_output_file)
            #print(ipwhois_time, file=header_output_file)
            #header_output_file.close()
            #print(url, file = urls_file)
        #counter = counter + 1        
    except Exception as e:
        print(e)
        Error=1
            #try:
            #    os.remove(dns_output_file_name)
            #    os.remove(ipwhois_output_file_name)
            #    os.remove(http_response_output_file_name)
            #    os.remove(header_output_file_name)
            #    os.remove(html_output_file_name)
            #    os.remove(whois_output_file_name)
            #except Exception as e:
            #    pass
    #urls_file.close()
    return html,dns_lookup_output, IPs, ipwhois, whois_output, content, domain, html_time, dns_lookup_time, ipwhois_time, Error
#main()

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
    data=download_url_content("http://docs.python-requests.org/en/master/user/quickstart/")
    print(data)
