'''
@author: lenovo
'''
import random
import pandas
import numpy as np
class Dyck2:
    '''
    one thing certained is that 's' and 'e' must be contained in. this len1 = 2.
    then shorten the strings randomly with 2x and pad 'e' in the end this len2 = length.
    so acutal_length = len1 + len2
    'e' is different with lambda(None) [1, 0, ..., 0] vs. [0, ... , 0]
    
    '''
    def padding_s(self, orign_s, length):
        alen = len(orign_s)
        need_pad = length - alen
        pad_s = 's' + orign_s + 'e' + 'l' * need_pad
        return pad_s
    def padding_l(self, orign_l, length):
        alen = len(orign_l)
        need_pad = length - alen
        if alen == 0:
            pad_l = '1' + '' + '1' + '1' * need_pad
        else:
            pad_l = '1' + orign_l + orign_l[-1] + orign_l[-1] * need_pad
        return pad_l
    def positive(self, length, variant):
        if variant:
            random.seed(random.randint(0, 100000))
            antipad = random.randint(0, length)
            #antipad = antipad if antipad % 2 == 0 else antipad - 1
        else:
            antipad = 0
        pairs = (length - antipad) // 2
        s = ""
        len = 0
        for i in range(pairs):
            ri = random.randint(0, len)
            bracket = "[]" if random.randint(0, 100) % 2 == 0 else "{}"
            s = s[: ri] + bracket + s[ri:]
            len = len + 2
        return s
    
    def positive_deepdepth(self, length, variant=False):
        pairs = length // 2
        D = {0: "[", 1: "{"}
        RD = {0: "]", 1:"}"}
        l = np.random.randint(0, 2, pairs).tolist()
        lr = l[:]
        lr.reverse()
        sl = "".join([D[i] for i in l])
        sr = "".join([RD[i] for i in lr])
        s = sl + sr
        return s
    def label_output(self, s, check):
        depth = 0
        stack = list()
        sl = list(s)
        output = list()
        for c in sl:

            if stack:
                if ( stack[-1] == "[" and c == "]" ) or ( stack[-1] == "{" and c == "}" ):
                    stack.pop()
                else:
                    stack.append(c)
            else:
                stack.append(c)
            if stack:
                output.append('0')
            else:
                output.append('1')
            depth = max(depth, len(stack))
        if check:
            return (True if output[-1] == '1' else False, depth)
        else:
            return (''.join(output), depth)
    def negative(self, length, variant):
        alen = 0
        while alen == 0:
            ps = self.positive(length, variant)
            alen = len(ps)
        
        sl = list(ps)
        while self.label_output(''.join(sl), True)[0]:
            rs = random.randint(0, alen-1)
            re = random.randint(rs, alen)
            ss = sl[rs: re]
            random.shuffle(ss)
            sl[rs: re] = ss
        return ''.join(sl)
def mm_ct_mo():
    trd_path = r'C:\Users\lenovo\git\StackNN\data\dyck2_train_30_1024_6_27'
    od = pd.read_csv(trd_path, header=None, dtype={0: str, 1:str})
    nl = list()
    length = len(od)
    slength = len(od.iloc[0, 0])
    print(length)
    for i in range(length):
        tmpl = od.iloc[i, :].tolist()
        eindex = tmpl[0].find("e")
        content = eindex + 1 - 2
        for j in range(content):
            tmp = (tmpl[0][:j+1] + "e" + "l" * (slength - 2 - j), tmpl[1][:j+1] + tmpl[1][j] * (slength - j - 1), True if tmpl[1][j] == "1" else False)
            nl.append(tmp)
    nl.append(("se" + "l" * (slength - 2), "1" * slength, True))
    print(len(nl))
    nl = list(set(nl))
    print(len(nl))
    newl = pd.DataFrame(nl)
    newl.to_csv(r'C:\Users\lenovo\git\StackNN\data\dyck2_train_30_1024_6_27formanytoone', header=None, index=None)
if __name__ == "__main__":
    # want unrepeat
    # variant(train) or in-variant(test)
    # deep
    
    D = Dyck2()
    len2 = 510
    len1 = 2
    actual_len = len1 + len2
    numbers = 512
    purpose = "test"
    depth_p = 0
    depth_n = 0
    variant = False if purpose == "test" else True
    sl = set()
    while len(sl) < numbers:
        ll = list()
        for i in range(numbers // 2):
            s = D.positive(len2, variant)
            s_label, depth_t= D.label_output(s, False)
            tmp = (D.padding_s(s, len2), D.padding_l(s_label, len2), True)
            ll.append(tmp)
            depth_p = max(depth_p, depth_t)
        for i in range(numbers // 2):
            s = D.negative(len2, variant)
            s_label, depth_t = D.label_output(s, False)
            tmp = (D.padding_s(s, len2), D.padding_l(s_label, len2), False)
            ll.append(tmp)
            depth_n = max(depth_n, depth_t)
        sl1 = set(ll)
        sl = sl.union(sl1)
    while len(sl) > numbers:
        sl.pop()
    sll = list(sl)
    random.shuffle(sll)
    ld_shuffled = pandas.DataFrame(sll)
    ld_shuffled.to_csv('dyck2_' + purpose + "_" + str(len2) + "_" + str(numbers) + "_" + str(depth_p) + "_" + str(depth_n), header=False, index=False)