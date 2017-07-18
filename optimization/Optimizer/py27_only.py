
import win32com.client
import json

def return_json(url):
    h = win32com.client.Dispatch('WinHTTP.WinHTTPRequest.5.1')
    h.SetAutoLogonPolicy(0)
    h.Open('GET', url, False)
    h.Send()
    try:
        js = json.loads(h.responseText)
    except ValueError:
        raise ValueError("The datastore did not repond with JSON: %s", url)
    if js['status'] != 0:
        raise ValueError("Data store returned status %s for url %s", js['response'], url)
    return js['response']


def download_options():
    url = 'http://rdpeds01.ortec.finance/ice/json/SPX%20index/IMPVOL_{}0.00%25'
    spx = {i:return_json(url.format(i)) for i in range(5,16)}
    with open(r'spx_option_data.json', 'w') as f:
        json.dump(spx, f)

def download_swaptions():
    url = 'http://rdpeds01.ortec.finance/ice-swaptions/json/USD/{}Y/{}Y/2017-06-30'
    spx = {i:return_json(url.format(i,10-i)) for i in [1,2,3,4,5,7]}
    with open(r'usd_swaption_data.json', 'w') as f:
        json.dump(spx, f)
