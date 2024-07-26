from sparkapi.config import SparkConfig
from sparkapi.core.chat.api import SparkAPI as ChatAPI
from sparkapi.core.image_generation.api import SparkAPI as ImageGenerationAPI
from sparkapi.core.image_understanding.api import SparkAPI as ImageUnderstandingAPI
import re

def extract_coordinates(coord_str):
    # 尝试匹配第一种格式
    match = re.search(r'\(x,y,z\) = \((-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)', coord_str)
    if match:
        x = float(match.group(1))
        y = float(match.group(2))
        z = float(match.group(3))
        return x, y, z
    
    # 尝试匹配第二种格式
    match = re.search(r'\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)', coord_str)
    if match:
        x = float(match.group(1))
        y = float(match.group(2))
        z = float(match.group(3))
        return x, y, z
    
    # 如果没有匹配到，返回None
    return None, None, None
# api = ChatAPI(**SparkConfig().model_dump())
# result = api.get_completion('你好')
# print(''.join(result))

# ImageGeneration
# api = ImageGenerationAPI(**SparkConfig(api_model='image_generation').model_dump())
# result = api.get_completion('帮我生成一张二次元风景图', outfile='out.png')
# print(result)

# # ImageUnderstanding
api = ImageUnderstandingAPI(**SparkConfig(api_model='image_understanding').model_dump())
result = api.get_completion('D:/workspace/api/img/t6.png', '返回图片上的文字内容')
# print(''.join(result))
answer1 = ''.join(result)
# print(answer1)
api = ChatAPI(**SparkConfig().model_dump())
result = api.get_completion(f'{answer1}, 计算机器人的坐标(x,y,z)的最终值, \
                            以$(x,y,z) = (x,y,z)$的固定格式给我,\
                            x为小数的形式,\
                            y为小数的形式,\
                            z为小数的形式,\
                            不带文字或多余的标点符号')
answer2 = ''.join(result)
# 输出answer2的倒数第二个元素
# print(answer2)
answer3 = answer2.split('$')
answer4 = answer3[-2]
# print(answer4)
x, y, z = extract_coordinates(answer4)
if x is not None:
    print(f'({x}, {y}, {z})')
else:
    print('未能识别出坐标值')
