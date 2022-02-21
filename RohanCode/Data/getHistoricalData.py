
import requests
def getData(symbol, data_range='30y', data_interval='1d'):

    url = 'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={data_range}&interval={data_interval}'.format(**locals())

    payload={}
    headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36"}

    response = requests.request("GET", url, headers=headers, data=payload)

    print(response.text)
getData('GOOGL')
