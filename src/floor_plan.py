from PIL import Image, ImageDraw

# 图像尺寸设置
image_size = 256  # 输出图像大小
room_width = 160  # 房间宽度（像素）
room_height = 120  # 房间高度（像素）
wall_thickness = 8  # 墙的厚度（像素）

# 计算房间在图像中的居中位置
start_x = (image_size - room_width) // 2
start_y = (image_size - room_height) // 2

# 创建空白图像（白色背景）
image = Image.new('L', (image_size, image_size), 0)
draw = ImageDraw.Draw(image)

# 绘制居中的实心矩形（黑色）
draw.rectangle([
    (start_x, start_y),
    (start_x + room_width - 1, start_y + room_height - 1)
], fill=255)

# 保存图像
image.save("floor_plan.png")

print("平面图已生成并保存为 'floor_plan.png'。")