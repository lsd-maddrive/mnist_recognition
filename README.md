# object_detection

## Содержание <a id="toc"></a>

* [Разработка](#develop)
    * [Через make](#make)
    * [Без make](#no_make)
    * [Как работать после установки poetry](#poetry_info)
* [Git Hook](#git-hook)
    * [Linux](#linux-hook)
    * [Windows](#windows-hook)
* [Соглашение по именованию коммитов](#commit-names)


## Разработка <a id="develop"></a>

### Через make <a id="make"></a>

* Подготовка окружения: `make prepare-project`

### Без make <a id="no_make"></a>

1. Устанавливаем [poetry](https://python-poetry.org/docs/#installation)
2. Ставим требуемые пакеты: `poetry install`

### Как работать после установки poetry <a id="poetry_info"></a>

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

[Содержание](#toc)

## Git Hook <a id="git-hook"></a>

> В make prepare-project он уже включён

* Добавить `git-hook`, чтобы в коммит автоматически добавлялось имя ветки.

Такой hook при каждом git-коммите будет автоматически добавлять в начало сообщения имя ветки в формате
```bash
[<branch_name>] <your commit message>
```

### Linux <a id="linux-hook"></a>

> В make prepare-project он уже включён

1) Сделать скрипт `git_hook/git-hook.sh` исполняемым `chmod +x git_hook/git-hook.sh`

2) Запустить скрипт (только один раз, чтобы создать hook) `./git_hook/git-hook.sh`

### Windows <a id="windows-hook"></a>

1) Запустить скрипт (только один раз, чтобы создать hook) `git_hook/git-hook.bat`

[Содержание](#toc)

## Соглашение по именованию коммитов <a id="commit-names"></a>

* [NF] - New Functionality/Features: новый функционал
* [CF] - Current Functionality: доработка существующего функционала
* [BF] - Bug Fix: исправление ошибок
* [RF] - ReFactoring: доработка кода, не затрагивающая бизнес-логику

[Содержание](#toc)
