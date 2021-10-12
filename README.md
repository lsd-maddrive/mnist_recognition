# object_detection

## Содержание <a id="toc"></a>

* [Разработка](#develop)
    * [Через make](#make)
    * [Без make](#no_make)

## Разработка <a id="develop"></a>

### Через make <a id="make"></a>

* `make prepare-project` - установка poetry и pre-commit

### Без make <a id="no_make"></a>

1. Установить [poetry](https://python-poetry.org/docs/#installation)
2. Запустить команду `poetry install`

Чтобы установить модуль библиотеки, используй команду:
```bash
poetry add <package_name> --dev
```

Чтобы удалить модуль библиотеки, используй команду:
```bash
poetry remove <package_name> --dev
```

Чтобы запустить скрипт внутри окружения poetry, есть 2 варианта:

1. `poetry run python3 script.py` - poetry автоматически запустит скрипт в своём окружении

2. с использованием двух команд:
    * `poetry shell` - активировать окружение poetry,
    * `python3 script.py` - запустить скрипт внутри окружения.
