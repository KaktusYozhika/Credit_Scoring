# Практика 4. Кредитный скоринг (группа 7)

Проект по дисциплине «Методология проектирования интеллектуальных платформ».
Группа БИВТ-23-АЭ-1, команда 7: Панютищева Екатерина, Кривоножкина Александра, Лаптева Екатерина.

Мы сделали сервис, который по кредитным характеристикам заёмщика оценивает риск серьёзной
просрочки (90+ дней) в следующие два года. Это продолжение практик 1-3, поэтому в коде
используются **Give Me Some Credit** и логистическая регрессия из практики 3,
а не пример данных по оттоку клиентов.

Система помогает выделить заявки на дополнительную проверку. Прогноз не означает
автоматического отказа в кредите. У модели нет подтверждённой калибровки вероятностей,
поэтому речь идёт об учебном прототипе.

## Состав проекта

| Файл или папка | Назначение |
|---|---|
| `data/cs-training.csv` | Данные проекта (доступ к Kaggle регулируется правилами соревнования) |
| `train.py` | Обучение модели одной командой и сохранение версий |
| `artifacts/versions/v1.0.0/` | Текущая модель, порог 0.55 |
| `artifacts/versions/v0.9.0/` | Предыдущая версия, порог 0.50 (для отката) |
| `app/main.py` | API `/health`, `/predict`, `/model-info`, `/docs` |
| `app/schemas.py` | Проверка входных данных через Pydantic |
| `tests/` | Тесты модели и запросов |
| `Dockerfile` | Сборка образа |
| `rollback_example.json` | Заявка, на которой видно изменение решения при откате |
| `requirements.txt` | Версии библиотек |

В каждой версии `artifacts/versions/.../` находятся `model.joblib`, `meta.json`
и `etalon.json` (пять сохранённых примеров для проверки результатов).

## Запуск из папки проекта

Рекомендуется Python 3.11. В терминале VS Code или Windows PowerShell:

```bash
python -m venv .venv
```

Активируйте окружение, если требуется:

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Установить зависимости и обучить модель можно так:

```bash
python -m pip install -r requirements.txt
python train.py
```

После этого в `artifacts/versions/` появятся две версии модели.
Скрипт использует тот же сплит 70/15/15, `random_state=42`, удаляет полные совпадения
без служебного индекса и обучает `Pipeline` из SimpleImputer, StandardScaler
и LogisticRegression. PR-AUC измеряется на validation, тестовая часть не участвует
в подборе модели и порога.

Запуск API (текущая версия):

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8077
```

Интерактивная страница: http://127.0.0.1:8077/docs

## Проверка API

Откройте в браузере http://127.0.0.1:8077/health. Вернётся `status: ok` и версия модели.

В PowerShell можно выполнить:

```powershell
Invoke-RestMethod http://127.0.0.1:8077/model-info
```

Пример `POST /predict` (PowerShell):

```powershell
$body = Get-Content -Raw -Encoding UTF8 example_request.json
Invoke-RestMethod -Uri http://127.0.0.1:8077/predict -Method Post -ContentType 'application/json' -Body $body
```

Сервис вернёт `probability`, `predicted_class`, `threshold`, `model_version`
и `recommendation`. `predicted_class=1` означает направление на дополнительную проверку.
При некорректных данных (например, `age=0`, отсутствии обязательного поля
или коде `NumberOfTimes90DaysLate=98`) API возвращает HTTP 422.
`MonthlyIncome` и `NumberOfDependents` необязательны (в датасете есть пропуски).

## Проверка тестами

```bash
python -m unittest discover -s tests -v
```

Тесты проверяют загрузку модели, метаданные, пять эталонных предсказаний,
запрос `/predict`, ошибки 422 и обе версии сервиса, включая изменение ответа
на заявке из `rollback_example.json`.

## Docker: сборка и запуск

Выполнить в корне проекта, где лежит Dockerfile:

```bash
docker build --build-arg MODEL_VERSION=v1.0.0 -t credit-scoring:1.0.0 .
docker run --rm -p 8077:8077 --name credit-scoring credit-scoring:1.0.0
```

В другом терминале снова открыть http://127.0.0.1:8077/health и
http://127.0.0.1:8077/model-info, затем отправить тот же POST `/predict`.
Результат должен совпадать с локальным API.

## Откат на предыдущую версию

В предыдущей версии такой же обученный Pipeline, но рабочий порог `0.50` вместо `0.55`.
Это позволяет показать возврат к ранее выбранной конфигурации без переобучения.

1. Остановить работающий контейнер (`Ctrl+C` или `docker stop credit-scoring`).
2. Собрать прошлую версию и запустить её:

```bash
docker build --build-arg MODEL_VERSION=v0.9.0 -t credit-scoring:0.9.0 .
docker run --rm -p 8077:8077 --name credit-scoring credit-scoring:0.9.0
```

3. Проверить `/model-info`: `model_version` станет `v0.9.0`, `threshold` будет `0.5`.
   Отправить `rollback_example.json`: класс станет `1` (на версии `v1.0.0` был `0`).
4. Для возврата на текущую версию остановить контейнер и снова запустить образ `credit-scoring:1.0.0`.

Локально предыдущую версию можно посмотреть без Docker:

Windows PowerShell:

```powershell
$env:MODEL_VERSION='v0.9.0'
python -m uvicorn app.main:app --port 8077
```

Linux/macOS:

```bash
MODEL_VERSION=v0.9.0 python -m uvicorn app.main:app --port 8077
```

## Ограничения данных

В `cs-training.csv` нет даты заявки и ID заёмщика, поэтому время сбора и повторы
одних клиентов определить нельзя. До сплита мы удаляем 609 полных совпадений по
признакам и целевой переменной без служебного индекса; совпадения не обязательно
означают одного человека. `MonthlyIncome` и `NumberOfDependents` частично пропущены,
а коды просрочек 96-98 требуют отдельной проверки; в API они отклоняются (422).

## Как сдать

По заданию практики 4 сдаётся **ссылка на Git-репозиторий**, а не только Word.
После проверки можно опубликовать проект в GitHub/GitLab и отправить ссылку преподавателю.
Важно проверить правила использования Kaggle: если распространение `cs-training.csv`
не разрешено, не загружать его в публичный репозиторий (файл уже добавлен в `.gitignore`),
а оставить инструкцию по скачиванию данных.

Пример команд для нового репозитория:

```bash
git init
git add .
git commit -m "Практика 4: кредитный скоринг как сервис"
git branch -M main
git remote add origin <АДРЕС_ВАШЕГО_РЕПОЗИТОРИЯ>
git push -u origin main
```
