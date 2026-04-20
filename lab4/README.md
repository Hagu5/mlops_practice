# Lab 4 — Версионирование датасета Titanic через DVC

Практика DVC: три версии датасета на базе `catboost.datasets.titanic()`,
каждая хранится в удалённом S3-хранилище (локальный MinIO), переключение
между версиями через `git checkout <tag> && dvc checkout`.

## Стек

| Компонент     | Выбор                               |
|---------------|--------------------------------------|
| Зависимости   | `uv` + `pyproject.toml`              |
| Датасет       | `catboost.datasets.titanic()` (train, 891 строка) |
| DVC remote    | MinIO (S3 API) в Docker              |
| Идентификация | access/secret key в `.dvc/config.local` (gitignore) |

### Почему не Google Drive

Изначально план был с `gdrive://` remote, но в 2024–2026 Google выключил
два возможных пути для личных Gmail-аккаунтов:

1. OAuth flow — приложение `dvc-gdrive` не прошло верификацию, Google
   блокирует его на этапе «Приложение заблокировано».
2. Service account — у SA нет квоты storage, загрузка отвергается:
   `Service Accounts do not have storage quota. Leverage shared drives
   or use OAuth delegation`. Shared Drives требуют Google Workspace.

Поэтому remote развёрнут как локальный MinIO через `docker compose` —
интерфейс полностью S3-совместимый, для курса эквивалентно «облаку».

## Структура

```
lab4/
├── pyproject.toml         # uv-проект: catboost, pandas, dvc[s3]
├── uv.lock
├── docker-compose.yml     # MinIO + создание bucket dvc-lab4
├── .dvc/
│   ├── config             # remote minio -> s3://dvc-lab4
│   ├── config.local       # (gitignored) access/secret для MinIO
│   └── .gitignore
├── .dvcignore
├── .gitignore             # исключает dvc4-*.json и .venv/
├── scripts/
│   ├── make_v1_raw.py     # v1: срез Pclass, Sex, Age
│   ├── make_v2_fillna.py  # v2: Age.fillna(mean)
│   └── make_v3_onehot.py  # v3: one-hot для Sex
├── data/
│   ├── .gitignore         # (сгенерирован dvc add) игнорит titanic.csv
│   ├── titanic.csv        # артефакт, под DVC
│   └── titanic.csv.dvc    # meta: MD5, размер, имя
└── README.md
```

## Воспроизведение с нуля

### 1. Запустить MinIO (remote)

```bash
cd lab4
docker compose up -d          # поднимет minio + создаст bucket dvc-lab4
```

Проверить: http://localhost:9001 (minioadmin / minioadmin).

### 2. Установить зависимости

```bash
uv sync
```

### 3. Прописать креды MinIO в локальный конфиг DVC

```bash
uv run dvc remote modify --local minio access_key_id minioadmin
uv run dvc remote modify --local minio secret_access_key minioadmin
```

### 4. Подтянуть данные из remote

```bash
uv run dvc pull
```

### 5. Переключение между версиями

```bash
# v1: Pclass, Sex, Age (NaN в Age присутствует)
git checkout lab4-v1 && uv run dvc checkout && head -4 data/titanic.csv

# v2: Age.fillna(mean = 29.70), NaN = 0
git checkout lab4-v2 && uv run dvc checkout && head -4 data/titanic.csv

# v3: one-hot для Sex → Sex_female, Sex_male
git checkout lab4-v3 && uv run dvc checkout && head -4 data/titanic.csv

git checkout master && uv run dvc checkout
```

## Версии датасета

| Tag       | Что сделано              | Shape     | MD5                                |
|-----------|--------------------------|-----------|------------------------------------|
| `lab4-v1` | Pclass, Sex, Age (raw)   | (891, 3)  | `9b176eeba7dc6fc2835649ba781b5700` |
| `lab4-v2` | Age.fillna(mean = 29.70) | (891, 3)  | `3b279155b9f20919e94046e7a59f57c1` |
| `lab4-v3` | one-hot encoding Sex     | (891, 4)  | `3b04621a903f9c6e7d5363b932279034` |

Выходные колонки:
- v1: `Pclass, Sex, Age`
- v2: `Pclass, Sex, Age` (Age без NaN)
- v3: `Pclass, Age, Sex_female, Sex_male`

## Log проверки checkout (реальный вывод)

```
=== v1 === HEAD: 3e6dae1
Pclass,Sex,Age
3,male,22.0
shape: (891, 3) | nan Age: 177

=== v2 === HEAD: 08922e9
Pclass,Sex,Age
3,male,22.0
shape: (891, 3) | nan Age: 0 | Age mean: 29.7

=== v3 === HEAD: 36dbee0
Pclass,Age,Sex_female,Sex_male
3,22.0,0,1
shape: (891, 4) | cols: ['Pclass','Age','Sex_female','Sex_male']
```

`dvc status -c` на `master` возвращает:
```
Cache and remote 'minio' are in sync.
```

## Где лежат артефакты

В MinIO bucket `dvc-lab4`:

```
files/md5/9b/176eeba7dc6fc2835649ba781b5700   # v1
files/md5/3b/279155b9f20919e94046e7a59f57c1   # v2
files/md5/3b/04621a903f9c6e7d5363b932279034   # v3
```

## Что не попадает в git

- `data/titanic.csv` — трекает DVC (см. `data/.gitignore`)
- `.dvc/config.local` — локальные креды (см. `.dvc/.gitignore`)
- `dvc4-*.json` — бывший ключ service account (см. `lab4/.gitignore`)
- `.venv/` — виртуалка uv
