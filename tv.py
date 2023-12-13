import re 
import math

def cleaned_features(featuresmap):
    for key, feature_value in featuresmap.items():
        inch_pattern = re.compile(r'(?<!\w)(\d+(?:\.\d+)?)(?:Inch(?:es)?|"\s*|-inch\s*| inch\s*|inch)(?!\w)', re.IGNORECASE)
        hertz_pattern = re.compile(r'\b(\d+)(?:Hertz|hertz|Hz|HZ|\shz\s*|-hz)\b', re.IGNORECASE)
        decimal_pattern = re.compile(r'(\d+)\.\d+ (\w+)')
            
        def inch_replace(match):
            numeric_value = float(match.group(1))
            rounded_value = math.ceil(numeric_value)
            return f'{rounded_value}inch'

        result = re.sub(inch_pattern, inch_replace, feature_value, re.IGNORECASE)
        result = re.sub(hertz_pattern, r'\1hz ', result, re.IGNORECASE)
        result = re.sub(decimal_pattern, r'\1\2', result)
        featuresmap[key] = result

        if key == 'Screen Size':
            if len(feature_value) <= 2:
                featuresmap[key] = feature_value + "inch"
            else: 
                match = re.search(r'(\d+(?:\.\d+)?)\s*inch|(\d+)\s*"\s*|\((\d+)\s*Diagonal size\)', feature_value, re.IGNORECASE)
                if match:
                    numeric_part = match.group(1) or match.group(2)
                    featuresmap['Screen Size'] = f"{numeric_part}inch"

        if key == 'ENERGY STAR Certified':
            if feature_value == 'Yes':
                featuresmap[key] = 'ESC1'
            if feature_value == 'No':
                featuresmap[key] = 'ESC0'

        if key == 'Width':
            width_class, no_class = find_width(feature_value)
            if no_class:
                featuresmap[key] = width_class

    if 'Width' in featuresmap:
        valid_feature = False
        for option in ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9']:
            if featuresmap['Width'] == option:
                valid_feature = True
                break
        if valid_feature == False:
            featuresmap.pop('Width')

    if 'Screen Size Class' not in featuresmap and 'Screen Size' in featuresmap:
        featuresmap['Screen Size Class'] = featuresmap.pop('Screen Size')

    if 'ENERGY STAR Certified' in featuresmap and featuresmap['ENERGY STAR Certified'] == 'Unknown':
        featuresmap.pop('ENERGY STAR Certified')
    return featuresmap


def find_width(feature):
    found = False
    match = re.search(r'\b(\d{2})\b', feature)
    if match: 
        found = True
        classwidth = 'class' + str(match.group(1))[0]
        return classwidth, found
    return None, found



class TV: 
    def __init__(self, tv_data):
        self.model_id = tv_data['modelID']
        self.features = cleaned_features(tv_data['featuresMap'])
        self.url = tv_data['url']
        self.title = self.cleaned(tv_data['title'])
        self.shop = tv_data['shop']
        self.brandname = None

    def __str__(self):
        string = f"ModelID: {self.model_id}, Title: {self.title}"
        return string
    
    def __eq__(self, other):
        if isinstance(other, TV):
            return self.model_id == other.model_id
        return False

    def get_shop(self):
        return self.shop

    def get_title(self):
        return self.title

    def get_features(self):
        return self.features
    
    def cleaned(self, title):
        inch_pattern = re.compile(r'\b(\d+)(?:Inch(?:es)?|"\s*|-inch\s*| inch\s*|inch)\b', re.IGNORECASE)
        hertz_pattern = re.compile(r'\b(\d+)(?:Hertz|hertz|Hz|HZ|\shz\s*|-hz)\b', re.IGNORECASE)
        result = re.sub(inch_pattern, r'\1inch ', title, re.IGNORECASE)
        result = re.sub(hertz_pattern, r'\1hz ', result, re.IGNORECASE)
        result = re.sub(r'-', '', result)
        return result
    

class TVList:
    def __init__(self, data):
        self.items = self.create_tv_list(data)

    def create_tv_list(self, data):
        tv_list = []
        for tvs_per_model in data.values():
            for tv in tvs_per_model:
                tv_object = TV(tv)
                tv_list.append(tv_object)
        return tv_list
    