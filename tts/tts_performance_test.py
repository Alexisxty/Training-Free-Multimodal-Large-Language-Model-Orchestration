import asyncio
import time
import os
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import numpy as np
import sys

# 添加项目根目录到路径，以支持导入
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tts.tts_manager import TTSManager
from tts.engines import TTSEngine
from utils.config import TEMP_FILE_CONFIG


async def test_tts_performance(
    text: str, 
    sentence_lengths: List[int], 
    repeat_times: int = 3,
    engine: TTSEngine = TTSEngine.COSYVOICE  # 默认使用CosyVoice
) -> Tuple[dict, dict, dict, dict]:
    """
    测试TTS合成性能对比：整体合成 vs 分段合成
    
    Args:
        text: 要合成的文本
        sentence_lengths: 要测试的不同句子长度列表
        repeat_times: 每个测试重复次数，取平均值
        engine: 要使用的TTS引擎
        
    Returns:
        whole_results: 整体合成的结果统计
        chunked_results: 分段合成的结果统计
        connection_results: 句子拼接时间的结果统计
        first_char_results: 首字符响应时间的结果统计
    """
    manager = TTSManager()
    
    # 设置使用CosyVoice引擎
    manager.set_engine(engine)
    print(f"[测试] 使用TTS引擎: {engine.value}")
    
    # 禁用实际播放
    manager.audio_player.is_playing_enabled = False
    
    # 创建测试结果存储目录
    test_dir = os.path.join(TEMP_FILE_CONFIG["TEMP_ROOT_DIR"], "performance_test")
    os.makedirs(test_dir, exist_ok=True)
    
    whole_results = {}
    chunked_results = {}
    connection_results = {}  # 句子拼接时间统计
    first_char_results = {}  # 首字符响应时间统计
    
    print("开始TTS性能测试...")
    
    # 对每种句子长度进行测试
    for length in sentence_lengths:
        whole_times = []
        chunked_times = []
        connection_times = []  # 句子拼接时间列表
        first_char_times_whole = []  # 整体合成的首字符响应时间
        first_char_times_chunked = []  # 分段合成的首字符响应时间
        
        print(f"\n测试句子长度: {length} 字符")
        
        # 准备文本
        if len(text) > length:
            test_text = text[:length]
        else:
            # 如果原文本不够长，则重复至所需长度
            repeat_count = (length // len(text)) + 1
            test_text = text * repeat_count
            test_text = test_text[:length]
        
        # 使用文本处理器拆分文本为句子
        sentences = manager.text_processor.split_text(test_text)
        
        print(f"- 文本长度: {len(test_text)} 字符")
        print(f"- 拆分为 {len(sentences)} 个句子")
        
        # 重复测试指定次数
        for i in range(repeat_times):
            # 清理缓存以确保准确测量首字符响应时间
            if i == 0:
                engine_id = manager.current_engine.get_engine_id()
                cache_path = manager.cache_manager.get_cache_path(test_text, engine_id)
                if os.path.exists(cache_path):
                    try:
                        os.remove(cache_path)
                        print(f"- 已清除缓存: {cache_path}")
                    except:
                        pass
                
                for sentence in sentences:
                    cache_path = manager.cache_manager.get_cache_path(sentence, engine_id)
                    if os.path.exists(cache_path):
                        try:
                            os.remove(cache_path)
                            print(f"- 已清除缓存: {cache_path}")
                        except:
                            pass
            
            # 测试1: 整体合成
            await manager.stop_speaking()  # 确保停止之前的播放
            manager.cleanup_all_temp_files()
            
            start_time = time.time()
            
            # 使用自定义标志捕获首字符响应时间
            first_char_detected = False
            first_char_time = 0
            
            # 子类化TTSManager来捕获首字符响应时间
            class TimingTTSManager(TTSManager):
                async def _process_and_queue_sentence(self, sentence, sentence_index):
                    nonlocal first_char_detected, first_char_time
                    if not first_char_detected:
                        first_char_time = time.time()
                        first_char_detected = True
                    await super()._process_and_queue_sentence(sentence, sentence_index)
            
            # 替换manager以使用我们的计时版本
            original_process_and_queue = manager._process_and_queue_sentence
            try:
                # 开始处理并测量首字符响应时间
                first_char_detected = False
                first_char_start = time.time()
                result_whole = await manager.text_to_speech(test_text)
                
                # 如果未通过回调检测到首字符时间，则使用当前时间作为近似值
                if not first_char_detected:
                    first_char_time = time.time()
                
                whole_time = time.time() - start_time
                first_char_response_whole = first_char_time - first_char_start
                
                whole_times.append(whole_time)
                first_char_times_whole.append(first_char_response_whole)
            finally:
                # 恢复原始方法
                manager._process_and_queue_sentence = original_process_and_queue
            
            # 保存整体合成的音频(仅第一次)
            if i == 0 and hasattr(manager, 'last_audio_file') and manager.last_audio_file:
                whole_audio_path = os.path.join(test_dir, f"whole_{length}_chars.wav")
                if os.path.exists(manager.last_audio_file):
                    import shutil
                    shutil.copy(manager.last_audio_file, whole_audio_path)
            
            # 测试2: 分段合成
            await manager.stop_speaking()  # 确保停止之前的播放
            manager.cleanup_all_temp_files()
            
            # 新增：分别测量每个句子的合成时间
            individual_sentence_times = []
            
            # 重置首字符检测状态
            first_char_detected = False
            first_char_start_chunked = time.time()
            
            for sentence in sentences:
                sentence_start_time = time.time()
                await manager.text_to_speech(sentence)
                sentence_time = time.time() - sentence_start_time
                individual_sentence_times.append(sentence_time)
                
                # 只捕获第一个句子的首字符响应时间
                if not first_char_detected:
                    first_char_time_chunked = time.time()
                    first_char_detected = True
            
            # 新增：计算总的合成时间和拼接开销
            total_sentence_time = sum(individual_sentence_times)  # 纯合成时间
            
            # 再次测试整个分段处理的总时间
            await manager.stop_speaking()
            manager.cleanup_all_temp_files()
            
            start_time = time.time()
            for sentence in sentences:
                await manager.text_to_speech(sentence)
            chunked_time = time.time() - start_time
            chunked_times.append(chunked_time)
            
            # 计算拼接开销时间
            connection_time = chunked_time - total_sentence_time
            connection_times.append(connection_time)
            
            # 计算分段模式的首字符响应时间
            first_char_response_chunked = first_char_time_chunked - first_char_start_chunked
            first_char_times_chunked.append(first_char_response_chunked)
            
            # 输出详细信息
            print(f"- 重复 {i+1}/{repeat_times}: 整体 {whole_time:.2f}秒, 分段总时间 {chunked_time:.2f}秒")
            print(f"  - 首字符响应时间: 整体 {first_char_response_whole:.3f}秒, 分段 {first_char_response_chunked:.3f}秒")
            if chunked_time > 0:
                connection_percent = connection_time/chunked_time*100
                print(f"  - 纯句子合成时间: {total_sentence_time:.2f}秒, 句子拼接开销: {connection_time:.2f}秒 ({connection_percent:.1f}%)")
            else:
                print(f"  - 纯句子合成时间: {total_sentence_time:.2f}秒, 句子拼接开销: {connection_time:.2f}秒 (无法计算百分比，分段总时间为0)")
        
        # 计算平均时间和标准差
        whole_avg = sum(whole_times) / len(whole_times)
        chunked_avg = sum(chunked_times) / len(chunked_times)
        connection_avg = sum(connection_times) / len(connection_times)
        first_char_whole_avg = sum(first_char_times_whole) / len(first_char_times_whole)
        first_char_chunked_avg = sum(first_char_times_chunked) / len(first_char_times_chunked)
        
        whole_std = np.std(whole_times) if len(whole_times) > 1 else 0
        chunked_std = np.std(chunked_times) if len(chunked_times) > 1 else 0
        connection_std = np.std(connection_times) if len(connection_times) > 1 else 0
        first_char_whole_std = np.std(first_char_times_whole) if len(first_char_times_whole) > 1 else 0
        first_char_chunked_std = np.std(first_char_times_chunked) if len(first_char_times_chunked) > 1 else 0
        
        # 计算性能损失百分比
        performance_loss = ((chunked_avg - whole_avg) / whole_avg) * 100 if whole_avg > 0 else 0
        
        # 计算拼接时间在总时间中的占比
        connection_percentage = (connection_avg / chunked_avg) * 100 if chunked_avg > 0 else 0
        
        # 计算首字符响应时间的差异百分比
        first_char_diff_percent = ((first_char_chunked_avg - first_char_whole_avg) / first_char_whole_avg) * 100 if first_char_whole_avg > 0 else 0
        
        whole_results[length] = {
            "avg_time": whole_avg,
            "std_dev": whole_std,
            "times": whole_times
        }
        
        chunked_results[length] = {
            "avg_time": chunked_avg,
            "std_dev": chunked_std,
            "times": chunked_times,
            "performance_loss": performance_loss
        }
        
        connection_results[length] = {
            "avg_time": connection_avg,
            "std_dev": connection_std,
            "times": connection_times,
            "percentage": connection_percentage
        }
        
        first_char_results[length] = {
            "whole_avg": first_char_whole_avg,
            "whole_std": first_char_whole_std,
            "chunked_avg": first_char_chunked_avg,
            "chunked_std": first_char_chunked_std,
            "whole_times": first_char_times_whole,
            "chunked_times": first_char_times_chunked,
            "diff_percent": first_char_diff_percent
        }
        
        print(f"- 平均时间: 整体 {whole_avg:.2f}秒, 分段 {chunked_avg:.2f}秒")
        print(f"- 性能损失: {performance_loss:.2f}%")
        print(f"- 拼接开销: {connection_avg:.2f}秒 (占分段合成总时间的 {connection_percentage:.1f}%)")
        print(f"- 首字符响应时间: 整体 {first_char_whole_avg:.3f}秒, 分段 {first_char_chunked_avg:.3f}秒")
        print(f"- 首字符响应时间差异: {first_char_diff_percent:.1f}%")
    
    # 恢复播放功能
    manager.audio_player.is_playing_enabled = True
    
    return whole_results, chunked_results, connection_results, first_char_results


async def generate_performance_report(
    results_whole: dict, 
    results_chunked: dict, 
    connection_results: dict,
    first_char_results: dict,
    output_dir: str
) -> None:
    """生成性能测试报告，包括图表"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取数据用于绘图
    lengths = sorted(list(results_whole.keys()))
    whole_times = [results_whole[l]["avg_time"] for l in lengths]
    chunked_times = [results_chunked[l]["avg_time"] for l in lengths]
    performance_losses = [results_chunked[l]["performance_loss"] for l in lengths]
    
    # 拼接时间数据
    connection_times = [connection_results[l]["avg_time"] for l in lengths]
    connection_percentages = [connection_results[l]["percentage"] for l in lengths]
    
    # 首字符响应时间数据
    first_char_whole_times = [first_char_results[l]["whole_avg"] for l in lengths]
    first_char_chunked_times = [first_char_results[l]["chunked_avg"] for l in lengths]
    first_char_diff_percentages = [first_char_results[l]["diff_percent"] for l in lengths]
    
    # 绘制时间对比图
    plt.figure(figsize=(12, 6))
    plt.plot(lengths, whole_times, 'b-o', label='整体合成')
    plt.plot(lengths, chunked_times, 'r-o', label='分段合成总时间')
    plt.plot(lengths, connection_times, 'g-o', label='句子拼接开销')
    plt.xlabel('文本长度(字符数)')
    plt.ylabel('处理时间(秒)')
    plt.title('TTS合成性能对比')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'tts_time_comparison.png'))
    
    # 绘制性能损失和拼接时间占比图
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(lengths, performance_losses)
    plt.xlabel('文本长度(字符数)')
    plt.ylabel('性能损失(%)')
    plt.title('分段合成的性能损失')
    plt.grid(True)
    
    # 绘制拼接时间占比图
    plt.subplot(1, 2, 2)
    plt.bar(lengths, connection_percentages)
    plt.xlabel('文本长度(字符数)')
    plt.ylabel('拼接开销占比(%)')
    plt.title('拼接开销在分段合成中的占比')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tts_overhead_analysis.png'))
    
    # 绘制首字符响应时间对比图
    plt.figure(figsize=(12, 10))
    
    # 首字符响应时间对比
    plt.subplot(2, 1, 1)
    plt.plot(lengths, first_char_whole_times, 'b-o', label='整体合成')
    plt.plot(lengths, first_char_chunked_times, 'r-o', label='分段合成')
    plt.xlabel('文本长度(字符数)')
    plt.ylabel('首字符响应时间(秒)')
    plt.title('TTS首字符响应时间对比')
    plt.legend()
    plt.grid(True)
    
    # 首字符响应时间差异百分比
    plt.subplot(2, 1, 2)
    plt.bar(lengths, first_char_diff_percentages)
    plt.xlabel('文本长度(字符数)')
    plt.ylabel('差异百分比(%)')
    plt.title('分段合成相比整体合成的首字符响应时间差异')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tts_first_char_response.png'))
    
    # 生成文本报告
    report_path = os.path.join(output_dir, 'tts_performance_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("TTS合成性能测试报告\n")
        f.write("===================\n\n")
        
        f.write("测试条件:\n")
        f.write(f"- 测试文本长度: {lengths}\n")
        f.write(f"- 每个长度重复测试次数: {len(results_whole[lengths[0]]['times'])}\n\n")
        
        f.write("测试结果:\n")
        f.write("--------\n\n")
        
        for length in lengths:
            f.write(f"文本长度: {length} 字符\n")
            f.write(f"- 整体合成平均时间: {results_whole[length]['avg_time']:.2f}秒\n")
            f.write(f"- 分段合成平均时间: {results_chunked[length]['avg_time']:.2f}秒\n")
            f.write(f"- 性能损失: {results_chunked[length]['performance_loss']:.2f}%\n")
            f.write(f"- 句子拼接开销: {connection_results[length]['avg_time']:.2f}秒\n")
            f.write(f"- 拼接开销占比: {connection_results[length]['percentage']:.1f}%\n")
            f.write(f"- 整体合成首字符响应时间: {first_char_results[length]['whole_avg']:.3f}秒\n")
            f.write(f"- 分段合成首字符响应时间: {first_char_results[length]['chunked_avg']:.3f}秒\n")
            f.write(f"- 首字符响应时间差异: {first_char_results[length]['diff_percent']:.1f}%\n\n")
        
        f.write("\n结论:\n")
        avg_loss = sum(performance_losses) / len(performance_losses)
        avg_connection = sum(connection_percentages) / len(connection_percentages)
        avg_first_char_diff = sum(first_char_diff_percentages) / len(first_char_diff_percentages)
        
        f.write(f"分段合成相比整体合成的平均性能损失为 {avg_loss:.2f}%\n")
        f.write(f"句子拼接平均占用分段合成总时间的 {avg_connection:.1f}%\n")
        f.write(f"分段合成的首字符响应时间相比整体合成平均 {avg_first_char_diff:.1f}% {'慢' if avg_first_char_diff > 0 else '快'}\n\n")
        
        if avg_connection > 30:
            f.write("建议: 拼接开销占比较高，可以考虑优化句子分段策略或使用整体合成方法。\n")
        else:
            f.write("建议: 拼接开销在可接受范围内，可以根据需求选择合适的合成方式。\n")
            
        if avg_first_char_diff < -10:  # 如果分段比整体快10%以上
            f.write("首字符响应时间建议: 对于需要快速响应的场景，分段合成可能更适合。\n")
        elif avg_first_char_diff > 10:  # 如果分段比整体慢10%以上
            f.write("首字符响应时间建议: 对于需要快速响应的场景，整体合成可能更适合。\n")
        else:
            f.write("首字符响应时间建议: 整体和分段合成在首字符响应速度上差异不明显。\n")
    
    print(f"性能报告已生成至: {output_dir}")


async def main():
    """主函数"""
    # 测试文本
    test_text = """
    人工智能(AI)是计算机科学的一个分支，它致力于开发能够执行通常需要人类智能的任务的机器。
    这些任务包括视觉感知、语音识别、决策制定和语言翻译。
    随着深度学习和神经网络的发展，AI技术在近年来取得了显著进步。
    语音合成是人工智能的一个重要应用领域，它允许计算机生成听起来自然的人类语音。
    """
    
    # 测试不同长度的文本
    sentence_lengths = [50, 100, 200, 300, 400]
    
    # 指定使用CosyVoice引擎
    engine = TTSEngine.COSYVOICE
    
    # 执行测试
    whole_results, chunked_results, connection_results, first_char_results = await test_tts_performance(
        text=test_text,
        sentence_lengths=sentence_lengths,
        repeat_times=3,
        engine=engine
    )
    
    # 生成报告
    report_dir = os.path.join(TEMP_FILE_CONFIG["TEMP_ROOT_DIR"], "performance_reports", engine.value)
    os.makedirs(report_dir, exist_ok=True)
    await generate_performance_report(whole_results, chunked_results, connection_results, first_char_results, report_dir)
    
    print(f"CosyVoice API测试完成，报告保存在: {report_dir}")


if __name__ == "__main__":
    asyncio.run(main()) 