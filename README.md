# MultiTankMaterialBalance

[![Build Status](https://github.com/sidelkin1/MultiTankMaterialBalance.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/sidelkin1/MultiTankMaterialBalance.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/sidelkin1/MultiTankMaterialBalance.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/sidelkin1/MultiTankMaterialBalance.jl)
[![DOI:10.1088/1755-1315/808/1/012034](https://img.shields.io/badge/DOI-10.1088%2F1755--1315%2F808%2F1%2F012034-blue)](https://doi.org/10.1088/1755-1315/808/1/012034)

В пакете реализована модель многоблочного материального баланса. Модель предназначена для прогноза пластового давления в блоках с учетом:

- Отборов жидкости и закачки воды через скважины внутри блоков
- Перетоков флюидов между соседними блоками из-за разности давления
- Притока воды из аквифера через блоки на внешней границе залежи

## Особенности пакета:

- Неявная численная схема для расчета пластового давления блоках на базе метода Ньютона
- Аналитическое вычисление якобиана
- Вычисление градиента относительно параметров блоков на базе метода сопряженных уравнений
- Реализация целевой функции с автоматической подгонкой коэффициентов продуктивности/приемистости скважин

## Интерактивный пример работы с пакетом

Для понимании теории и практики работы с пакетом создан [обучающий пример](https://github.com/sidelkin1/multitank-matbal-tutorial) на базе Jupyter Notebook.

## Интерфейс командной строки (CLI)

Для работы с пакетом реализован [интерфейс командной строки](https://github.com/sidelkin1/multitank-matbal-cli) (CLI). В нем реализована автоматическая настройки параметров блоков.

## Цитирование

Для цитирования использования пакета `MultiTankMaterialBalance` используйте следующие данные

```tex
@article{article,
author = {Sidelnikov, K and Tsepelev, V and Kolida, A and Faizullin, R},
year = {2021},
month = {07},
pages = {012034},
title = {Reservoir energy management based on the method of multi-tank material balance},
volume = {808},
journal = {IOP Conference Series: Earth and Environmental Science},
doi = {10.1088/1755-1315/808/1/012034}
}
```