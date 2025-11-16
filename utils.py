def is_float(s):
    s = s.replace("*","").replace("/","").replace("-","0")
    try:
        float(s)
        return True
    except ValueError:
        return False