# FedAvg with homomorphic encryption

项目架构

- Client.py: client定义
- main.py: 程序主入口
- model.py: 定义神经网络结构
- options.py: 可选选项
- utils.py: 辅助函数

运行命令示例

```bash
python3 main.py --cuda --mode=fedavg
```

具体参数见 options.py