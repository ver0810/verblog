---
title: Hadoop 伪分布式安装步骤
pubDatetime: 2025-10-10T04:06:31Z
slug: example-draft-post
featured: false
draft: false
tags:
  - big-data
  - hadoop
description:
  介绍如何在Ubuntu中安装Hadoop，并创建伪分布式节点。
---

本文使用VMware虚拟机安装。使用到的Ubuntu版本为最新版长期支持版本lst-24。

虚拟机安装系统过程略过。

## 创建hadoop用户

1. 创建hadoop用户
```bash
useradd -m hadoop -s /bin/bash
```

2. 设置hadoop密码
```bash
sudo passwd hadoop
```

3. 添加hadoop用户到sudo用户组
```bash
sudo adduser hadoop sudo
```

## 安装ssh并配置ssh无密码登录

切换到hadoop用户进行配置。
1. 安装ssh
```bash
# 更新软件包
sudo apt update

# 安装ssh
sudo apt install openssh-server
```

2. 生成ssh密钥
```bash
ssh-keygen -t rsa
```

3. 将公钥添加到已认证中
```bash
cd ~/.ssh

cat ./id_rsa.pub >> ./authorized_keys
```

4. 本地登录
```bash
ssh localhost
```

## 安装Java环境

1. 安装jdk
```bash
# 更新软件包
sudo apt update

sudo apt install openjdk-8-jdk
```

2. 添加环境变量到.bashrc
```bash
# 在.bashrc 中添加, 后面为jdk的安装路径
export JAVA_HOME=/usr/lib/jvm/jave-8-openjdk-amd
```

3. 更新.bashrc
```bash
source ~/.bashrc
```

## 安装hadoop

1. 下载hadoop
到apache hadoop 官网下载hadoop。
```bash
# 示例：下载 Hadoop 3.3.6
wget https://dlcdn.apache.org/hadoop/common/hadoop-3.3.6/hadoop-3.3.6.tar.gz
```

2. 解压并安装

安装到`/usr/local`文件夹方便管理。
```bash
# 解压到 /usr/local 目录
sudo tar -zxf hadoop-3.3.6.tar.gz -C /usr/local

# 进入 /usr/local 目录
cd /usr/local

# 重命名文件夹
sudo mv ./hadoop-3.3.6/ ./hadoop

# 修改文件夹所有者为 hadoop 用户
sudo chown -R hadoop:hadoop ./hadoop
```

## 格式化NameNode并启动hadoop

1. 格式化NameNode
```bash
hdfs namenode -format
```

2. 启动hadoop
```bash
cd /usr/local/hadoop

# 启动
./sbin/start-dfs.sh

# 验证启动
jps
```




