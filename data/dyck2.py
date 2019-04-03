'''
@author: lenovo
'''
import random
import pandas
import numpy as np
import nltk
print("numpy imported.")
class Dyck2:
    '''
    one thing certained is that 's' and 'e' must be contained in. this len1 = 2.
    then shorten the strings randomly with 2x and pad 'e' in the end this len2 = length.
    so acutal_length = len1 + len2
    'e' is different with lambda(None) [1, 0, ..., 0] vs. [0, ... , 0]
    
    '''
    def padding_s(self, orign_s, length=16):
        alen = len(orign_s)
        need_pad = length - alen
        pad_s = 's' + orign_s + 'e' + 'l' * need_pad
        return pad_s
    def padding_l(self, orign_l, length=16):
        alen = len(orign_l)
        need_pad = length - alen
        if alen == 0:
            pad_l = '0' + '' + '1' + '1' * need_pad
        else:
            pad_l = '0' + orign_l + orign_l[-1] + orign_l[-1] * need_pad
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
    def label_output(self, s, check=True):
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
    def negative(self, length):
        alen = 0
        while alen == 0:
            ps = self.positive_deepdepth(length, False)
            alen = len(ps)
        
        sl = list(ps)
        while self.label_output(''.join(sl))[0]:
            rs = random.randint(0, alen-1)
            re = random.randint(rs, alen)
            ss = sl[rs: re]
            random.shuffle(ss)
            sl[rs: re] = ss
        return ''.join(sl)

if __name__ == "__main__":
    # want unrepeat
    # variant(train) or in-variant(test)
    # deep
    
    D = Dyck2()
    len2 = 1022
    len1 = 2
    actual_len = len1 + len2
    numbers = 10
    purpose = "test"
    depth_p = 0
    depth_n = 0
    sl = set()
    while len(sl) < numbers:
        ll = list()
        for i in range(numbers // 2):
            s = D.positive_deepdepth(len2, False)
            s_label, depth_t= D.label_output(s, False)
            tmp = (D.padding_s(s, length=len2), D.padding_l(s_label, length=len2), True)
            ll.append(tmp)
            depth_p = max(depth_p, depth_t)
        for i in range(numbers // 2):
            s = D.negative(len2)
            s_label, depth_t = D.label_output(s, False)
            tmp = (D.padding_s(s, length=len2), D.padding_l(s_label, length=len2), False)
            ll.append(tmp)
            depth_n = max(depth_n, depth_t)
        #ld_shuffled = pandas.DataFrame(sklearn.utils.shuffle(ll))
        #ld_shuffled.to_csv('dyck2_train1minix10', header=False, index=False)
        sl1 = set(ll)
        sl = sl.union(sl1)
    while len(sl) > numbers:
        sl.pop()
    sll = list(sl)
    random.shuffle(sll)
    ld_shuffled = pandas.DataFrame(sll)
    ld_shuffled.to_csv('dyck2_' + purpose + "_" + str(len2) + "_" + str(numbers) + "_" + str(depth_p) + "_" + str(depth_n), header=False, index=False)