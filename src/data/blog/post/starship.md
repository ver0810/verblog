---
title: starship 安装教程
pubDatetime: 2025-09-02T00:00:00Z
slug: post-1
featured: false
draft: false
tags:
  - shell
  - Linux
description:
  starship 是一个使用rust编写的客制化shell的工具。
---

## starship 简介

轻量、迅速、客制化shell 模板，可以用于fish，zsh 等shell。

[官方文档](https://starship.rs/zh-CN/guide/)


## 安装

### 前置要求

需要安装nerd-font字体

### 安装starship

```
curl -sS https://starship.rs/install.sh | sh
```

### 配置shell

这里我使用fish， 在在`~/.config/fish/config.fish`的最后，添加以下内容
```
starship init fish | source
```


