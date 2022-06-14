# MNIST Recognition

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Система распознавания рукописных цифр из датасета MNIST с исследованием влияния состязательных атак на качество работы моделей.

[Подробное описание](./docs/mnist_description.md)

## Содержание

- [Содержание](#содержание)
- [Начало работы](#начало-работы)
- [Git Hook](#git-hook)
- [Соглашение по именованию коммитов](#соглашение-по-именованию-коммитов)
- [FAQ](#faq)
  - [Как работать после установки poetry](#как-работать-после-установки-poetry)


## Начало работы

1. Установка `make`
    * Windows

        Установить [chocolatey](https://chocolatey.org/install), после чего установить `make` с помощью команды:

        ```powershell
        choco install make
        ```

    * Linux

        ```bash
        sudo apt-get install build-essential
        ```

2. Установка `python 3.8`
    * Windows

        Установить через [официальный установщик](https://www.python.org/downloads/release/python-3810/)

    * Linux

        ```bash
        sudo apt install python3.8-dev
        ```

3. Установка [`poetry`](https://python-poetry.org/docs/#installation)
    * Windows

        Используйте [официальные инструкции](https://python-poetry.org/docs/#windows-powershell-install-instructions) или команду `powershell`

        ```powershell
        (Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python -
        ```

    * Linux

        ```bash
        make poetry-download
        ```
4. Установка требуемых пакетов (библиотек) и инструментов с помощью команды

    ```bash
    make install
    ```

5. Установка [Git LFS](https://git-lfs.github.com/) с помощью команды

    ```bash
    git lfs install
    ```

6. Скачать файлы из LFS

    ```bash
    git lfs pull
    ```

[Содержание](#содержание)


## Git Hook

При успешном выполнении команды `make install` будет установлен git hook, который будет автоматические добавлять в коммит имя ветки

То есть коммит в истории гита будет выглядить так:

```bash
[<branch_name>] <your commit message>
```

При этом имя ветки и скобки писать не нужно.

[Содержание](#содержание)


## Соглашение по именованию коммитов

Чтобы история гита была структурированной, принято использовать "маркеры", чтобы помечать изменения в коммите.

* [NF] - New Functionality/Features: новый функционал
* [CF] - Current Functionality: доработка существующего функционала
* [BF] - Bug Fix: исправление ошибок
* [RF] - ReFactoring: доработка кода, не затрагивающая бизнес-логику

Например, при рефакторинге кода коммит будет выглядеть следующим образом:

```bash
git commit -m "[RF]: minor refactoring of training script"
```

[Содержание](#содержание)


## FAQ

### Как работать после установки poetry

* Если хочется, чтобы для poetry работали автодополнения через Tab, то действуем по [инструкции](https://python-poetry.org/docs/master#enable-tab-completion-for-bash-fish-or-zsh)

* Чтобы установить модуль библиотеки, используй команду:
```bash
poetry add <package_name> --dev
```

* Чтобы удалить модуль библиотеки, используй команду:
```bash
poetry remove <package_name> --dev
```

* Чтобы запустить скрипт есть 2 пути:
    * Первый способ:
        1. Запустить (активировать) виртуальное окружение `poetry shell`
        2. Запускать скрипты (внутри окружения) привычным способом `python3 script.py`
    * Второй способ - использовать команду `poetry run python3 script.py`
    > (poetry сам свяжет с виртуальным окружением)
* Чтобы деактивировать виртуальное окружение - `poetry exit`

[Содержание](#содержание)
