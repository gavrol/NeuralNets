this is a description of how some of the data files in this dir have been created
1) test01.csv
    size = 1000
    vec1 = np.random.uniform(-0.5, 0.5,size)
    vec2 = np.random.uniform(-0.5, 0.5,size)
    target = vec1+vec2
    
2) test02_logExp.csv:
    vec1 = np.random.uniform(1,10,size)
    vec2 = np.random.uniform(-1,1,size)
    vec3 = np.random.uniform(-5,5,size)
    
    tvec1 = np.log(vec1)
    tvec2 = np.exp2(vec2)
    target = np.c_[tvec1+tvec2+vec3].reshape(size,1)

3) test03.csv:
    vec1 = np.linspace(-10,10,size)
    vec2 = np.linspace(-1,1,size)
    vec3 = np.linspace(-5,5,size)
    
    tvec1 = vec1 + np.random.uniform(-0.05,0.05,size)
    target = np.c_[tvec1+vec2+vec3].reshape(size,1)

4) test04.csv: 
	vec1 = np.random.uniform(-1,1,size)
    vec2 = np.random.uniform(-1,1,size)
    vec3 = np.random.uniform(-1,1,size)
    tvec1 = 2.5*(vec1)
    tvec2 = vec2 - np.ones(len(vec2))
    target = tvec1+tvec2+vec3

5) test05.csv:
    size = 1000
    vec1 = np.random.uniform(-5,5,size)
    vec2 = np.random.uniform(-5,5,size)
    vec3 = np.random.uniform(-5,5,size)
    target = np.tanh(vec1+vec2+vec3)

6)