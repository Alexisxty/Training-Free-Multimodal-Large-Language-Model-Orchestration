"""
PyQT-VITA 主程序入口

这个文件是项目的主入口点，根据命令行参数启动不同的模块：
- GUI模式：启动图形界面
- CLI模式：启动命令行界面
"""

import sys
import argparse
import asyncio

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PyQT-VITA 人工智能对话系统')
    parser.add_argument('--mode', type=str, choices=['gui', 'cli'], default='gui',
                      help='运行模式: gui (图形界面) 或 cli (命令行界面)')
    return parser.parse_args()

async def run_gui_mode():
    """运行GUI模式"""
    from gui.gui import main
    await main()

async def run_cli_mode():
    """运行CLI模式"""
    from dialogue.dialogue_system import DialogueSystem
    
    # 创建对话系统
    dialogue_system = DialogueSystem()
    
    print("CLI模式已启动，按Ctrl+C退出")
    
    # 运行对话系统
    await dialogue_system.run()

async def main():
    """主函数"""
    args = parse_args()
    
    if args.mode == 'gui':
        await run_gui_mode()
    else:
        await run_cli_mode()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序已退出")