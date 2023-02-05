from symato_dict import SYMATO_MARKTONES, SYMATO_SYMS
import bogo

'''
2816 vocab, `u16` type:
- `000 ... 255`:  255 bytes
- `256 ... 273`:   18 marktones
- `274 ...2807`: 2534 syms
- `2808...2815`:
  - `2815`: '^' viết hoa chữ cái đầu của sym
  - `2814`: '^^' viết hoa toàn bộ sym
'''

marktones = SYMATO_MARKTONES.split()
assert len(marktones) == 18

syms = SYMATO_SYMS.split()
assert len(syms) == 2534

class Symato:
	def is_sym_capitalized(self, token_id): return token_id == 2814
	def is_capitalized(self, token_id): return token_id == 2814 or token_id == 2815
	def is_byte(self, token_id): return token_id <= 255
	def is_marktone(self, token_id): return 256 <= token_id and token_id <= 273
	def is_sym(self, token_id): return 274 <= token_id and token_id <= 2807
	def vocab_size(self): return 2816

	def to_utf8(self, sym_id, marktone_id):
		return bogo.core.process_sequence(self.itos[sym_id] + self.itos[marktone_id][1:])

	def tids_to_utf8(self, tids):
		i, n = -1, len(tids) - 1
		str = ""; capitalized_tid = None
		while i < n:
			i += 1
			tid = tids[i]

			if self.is_capitalized(tid):
				capitalized_tid = tid
			else:
				token = None
				if self.is_sym(tid) and i < n - 1:
					mtid = tids[i + 1]
					if self.is_marktone(mtid):
						token = self.to_utf8(tid, mtid)
						i += 1
				else:
					token = self.decode(tid)

				if capitalized_tid != None:
					if self.is_sym_capitalized(capitalized_tid): token = token.upper()
					else: token = token[0].upper() + token[1:]
					capitalized_tid = None
				str += token

		return str

	def __init__(self):
		self.stoi = {"^": 2815, "^^": 2814}
		self.itos = {2815: "^", 2814: "^^"}

		for i, s in enumerate(marktones):
			sid = i + 256
			self.stoi[s] = sid
			self.itos[sid] = s
		assert self.stoi["|zj"] == 273
		assert self.itos[256] == "|"

		for i, s in enumerate(syms):
			sid = i + 274
			self.stoi[s] = sid
			self.itos[sid] = s
		assert self.stoi["yeu"] == 2807
		assert self.itos[274] == "a"
	
	def ids_to_tokens(self, ids):
		return [self.decode(i) for i in ids]
	
	def decode(self, tid):
		return self.itos[tid] if tid > 255 else chr(tid)

	def encode(self, a, maxx=None):
		if isinstance(a, str): a = bytes(a, "utf8")
		b, e = 0, len(a)
		if maxx is None: maxx = e
		if maxx > e: maxx = e
		tids = []
		while (b < maxx):
			c = a[b]
			if c == 16:
				n = b + 1
				while a[n] != 16: n += 1
				s = str(a[b+1:n], "utf8")
				# print(s)
				if s[0] == '^':
					if s[1] == '^':
						tids.append(self.stoi["^^"])
						s = s[2:]
					else:
						tids.append(self.stoi["^"])
						s = s[1:]
				tokens = s.split("|")
				tids.append(self.stoi[tokens[0]])
				if len(tokens) > 1:
					tids.append(self.stoi["|" + tokens[1]])

				b = n + 1
			else:
				tids.append(c)
				b += 1
		a = None
		return tids

	def tokenize(self, filename, maxx=None):
		return self.encode(open(filename,"rb").read(), maxx)


# '''
# s = Symato()
# print(s.ids_to_tokens([2815, 2648, 273, 32, 2814, 1487, 256]))
# print(s.encode("^quyen|zf luc|wj cua|r ong|z trum|f xa|x hoi|zj dden| (^ky|f 3): ^bi| kich|j gia| ddinh|f.", 500))
# tids = [32, 66, 79, 84, 32, 84, 50, 46, 32, 10, 104, 116, 116, 112, 115, 58, 47, 47, 118, 110, 101, 120, 112, 114, 101, 115, 115, 46, 110, 101, 116, 47, 99, 100, 99, 45, 2418, 256, 45, 725, 256, 45, 116, 104, 97, 112, 45, 725, 256, 45, 547, 256, 45, 52, 50, 57, 57, 54, 50, 48, 46, 104, 116, 109, 108, 32, 10]
# print("".join(s.ids_to_tokens(tids)))
# tids = [2814, 2535, 267, 32, 2814, 645, 273, 32, 2814, 2249, 265, 32, 2814, 1340, 272, 32, 2814, 1893, 257, 32, 2814, 278, 268, 32, 2814, 1166, 268, 32, 2814, 1671, 273, 32, 2814, 1668, 264, 44, 32, 2814, 524, 256, 32, 2814, 681, 272, 32, 2814, 1666, 270, 58, 32, 10, 594, 261, 32, 1557, 268, 32, 1421, 269, 32, 2553, 262, 32, 2814, 2536, 256, 32, 2371, 259, 32, 2265, 264, 32, 885, 257, 32, 1656, 268, 32, 2814, 1656, 272, 32, 2410, 269, 32, 2804, 269, 32, 10, 10, 2814, 330, 259, 32, 2814, 2023, 256, 32, 2814, 1340, 272, 32, 2814, 1893, 257, 32, 2814, 278, 268, 32, 2814, 1671, 273, 32, 2814, 1668, 264, 58, 32, 10, 75, 104, 117, 225, 186, 175, 110, 103, 32, 110, 198, 176, 97, 110, 32, 110, 117, 196, 131, 110, 32, 10, 10, 2814, 330, 259, 32, 2814, 2023, 256, 32, 2814, 1340, 272, 32, 2814, 1892, 257, 32, 2814, 278, 268, 32, 2814, 681, 272, 32, 2814, 1666, 270, 32, 2814, 2665, 263, 32, 2814, 2411, 269, 32, 2814, 281, 256, 58, 32, 10, 89, 101, 110, 44, 32, 84, 105, 101, 110, 32, 98, 117, 111, 110, 103, 32, 98, 111, 111, 109, 32, 10, 10, 45, 32, 45, 32, 45, 32, 10, 2815, 964, 256, 32, 112, 104, 117, 99, 32, 1283, 256, 32, 2316, 256, 32, 2615, 256, 32, 121, 101, 110, 32, 2418, 256, 32, 1127, 256, 32, 2713, 256, 32, 1738, 256, 32, 846, 256, 32, 2316, 256, 32, 692, 256, 32, 2615, 256, 32, 1283, 256, 32, 99, 104, 117, 99, 32, 2316, 256, 32, 2238, 256, 32, 1006, 256, 32, 2311, 256, 32, 117, 111, 99, 32, 1656, 268, 32, 274, 256, 32, 10, 2815, 1340, 256, 32, 1787, 256, 32, 547, 256, 32, 771, 256, 32, 1787, 256, 32, 2478, 268, 32, 681, 256, 32, 116, 104, 117, 111, 110, 103, 32, 2023, 256, 32, 2627, 256, 32]
# print(s.tids_to_utf8(tids))
# '''