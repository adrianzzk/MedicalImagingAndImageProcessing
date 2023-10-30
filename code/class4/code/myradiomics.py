import SimpleITK as sitk
import radiomics

# 加载医学影像和标签

image = sitk.ReadImage('data/brain_image.nrrd')
label = sitk.ReadImage('data/brain_label.nrrd')

# 创建 Radiomics 提取器
extractor = radiomics.featureextractor.RadiomicsFeatureExtractor()

# 提取影像组学特征
result = extractor.execute(image, label)

# 输出结果
for key, value in result.items():
    print(f"{key}: {value}")
