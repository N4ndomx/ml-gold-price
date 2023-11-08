
import requests
body = {
    
  "year": 2016,
  "mileage": 40145,
  "city": 1769.0,
  "state": 44.0,
  "make": 15.0

    }
response = requests.post(url = 'http://127.0.0.1:8000/score',
              json = body)
print (response.json())
# output: {'score': 0.866490130600765}