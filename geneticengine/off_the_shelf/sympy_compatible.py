


def fix_logs(s: str):
    s = s.replace("np.log(1 + np.abs","log(")
    return s

def fix_sqrts(s: str):
    s = s.replace("np.sqrt(np.abs(","sqrt(")
    return s

def fix_numpy(s: str):
    s = s.replace("np.","")
    return s
    

def fix_all(s: str):
    s = fix_numpy(fix_sqrts(fix_logs(s)))
    return s