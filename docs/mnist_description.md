# Документация

## Содержание <a id="toc"></a>

* [Описание данных MNIST](#data_description)
* [Описание решаемой задачи](#task_description)
* [Описание модели](#model_description)

## Описание данных MNIST <a id="data_description"></a>

База данных [MNIST](http://yann.lecun.com/exdb/mnist/) (Модифицированная база данных Национального института стандартов и технологий) представляет собой большую коллекцию рукописных цифр.

Датасет содержит:
* обучающий набор из 60,000 примеров
* тестовый набор из 10,000 примеров

Данные представляют собой монохромные, нормализованые по размеру и центрированые изображения рукописных цифр. Каждая картинка имеет размер 28х28 пикселей (всего 784 пикселей).

Для дальнейшей работы с данными обучающий набор дополнительно разделён на 2 части: обучающую и валидационную выборки.
* Для обучения используется 70% обучающего набора т.е 42,000 изображений.
* Для валидации используется 30% обучающей выборки т.е 18,000 изображений.

[Содержание](toc)

## Описание решаемой задачи <a id="task_description"></a>

**Поставленная задача** - это задача распознавания рукописных цифр. Датасет состоит из 10 различных классов в диапазоне от 0 до 9, т.е. решаемая задача - это многоклассовая классификация (multiclass classification).

[Содержание](toc)

## Описание модели <a id="model_description"></a>

**Архитектура модели**: полносвязный перцептрон

<p align="center">
<img src="./assets/model_architecture.png">
</p>

**Функция потерь**: [`nn.CrossEntropyLoss`](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#crossentropyloss)

**Оптимизатор**: [`torch.optim.Adam`](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam)

**"Планировщик скорости обучения"** - [`ReduceLROnPlateau`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html)

<center> Графики функции потерь и изменения скорости ообучения </center>

|Train Loss                         | Valid Loss                        | Learning Rate                     |
|:---------------------------------:|:---------------------------------:|:---------------------------------:|
|![Train](./assets/train_loss.jpg)  |![Valid](./assets/valid_loss.jpg)  |![LR](./assets/learning_rate.jpg)  |

[Содержание](toc)
