import gdown
url = 'https://drive.google.com/file/d/1l6aofWUSNDZoT1hl82hiGuu_SNOz5tHi/view?usp=sharing'
output = 'intervals_rolling_dataframe0.csv'
gdown.download(url, output, quiet=False)

url = 'https://drive.google.com/file/d/1q4bUkLPa8TLH128Umu6I2FIGB3ROWV2y/view?usp=sharing'
output = 'test_data.csv'
gdown.download(url, output, quiet=False)