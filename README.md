# MWS Model ML

Коротко про проект:
На вход поступает json вида: 
```
[
  {
    "id": 1,
    "rule_id": "aws-access-key",
    "file_path": "src/example/config.py",
    "line": 10,
    "value": "AKIAIOSFODNN7EXAMPLE",
    "severity": "warning",
    "scanner_confidence": 0.65
  },
  {
    "id": 2,
    "rule_id": "private-key",
    "file_path": "src/keys/prod.pem",
    "line": 15,
    "value": "-----BEGIN PRIVATE KEY-----",
    "severity": "error",
    "scanner_confidence": 0.92
  }
]
```

Запуск сервера происходит таким образом:

>docker run -p 5000:5000 -v ${PWD}/models:/app/models fpni9255/mws_model_ml





На выходе выводится MLPredict и MLConfidence для каждого ID соответственно.  