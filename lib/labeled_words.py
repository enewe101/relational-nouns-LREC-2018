class LabeledWords(list):

	subtypes = {
		'b': 'body-part',
		'r': 'link',
		'a': 'aspectual',
		'f': 'functional',
		'v': 'deverbal',
		'p': 'portion',
		'j': 'adjectival',
	}


	def __init__(self, path=None):

		super(LabeledWords, self).__init__()
		if path is not None:
			self.read(path)


	def get(self, typestring):

		if len(typestring) < 3:
			raise ValueError(
				'The typestring must have three characters:\n'
					'\t1st: whether is partial\n'
						'\t\t[*ms] (= either, partial, strict)\n'
					'\t2nd: whether is relational\n'
						'\t\t[*pn] (= either, relational, non-relational)\n'
					'\t3rd: subtype\n'
						'\t\t[*bpjvafr]+ (see subtypes.  * means any)\n'
			)
		is_partial = typestring[0]
		is_relational = typestring[1]
		subtype = typestring[2:]

		return_set = set()
		for word in self:

			if is_partial != '*':
				if is_partial == 'm' and not word['is_partial']:
					continue
				elif is_partial == 'n' and word['is_partial']:
					continue

			if is_relational != '*':
				if is_relational == 'p' and not word['is_relational']:
					continue
				elif is_relational == 'n' and word['is_relational']:
					continue

			if subtype != '*':
				if word['subtype-code'] not in subtype:
					continue

			return_set.add(word['word'])

		return return_set


	def read(self, path):

		for line in open(path):
			line = line.strip()
			if len(line) < 1 or line.startswith('#'):
				continue
			self.append(self.read_word(line))


	def read_word(self, line):
		word, typestring = line.strip().split()
		original_typestring = typestring
		is_partial = False

		# Is the first character an "m", which stands for "mainly"
		if typestring.startswith('m'):
			is_partial = True
			typestring = typestring[1:]

		# Is the next character an "n" or a "p" (negative or positive)
		if typestring.startswith('p'):
			is_relational = True
		elif typestring.startswith('n'):
			is_relational = False
		else:
			raise ValueError(
				'No relational indicator: %s.' % original_typestring)

		typestring = typestring[1:]
		if len(typestring) == 0:
			subtype_code = 'n'
			subtype = None

		elif len(typestring) == 1:
			subtype_code = typestring
			try:
				subtype = self.subtypes[subtype_code]
			except KeyError:
				raise ValueError(
					'Unrecognized noun subtype: %s.' % original_typestring
				)

		else:
			raise ValueError(
					'Trailing characters on typestring: %s.' 
					% original_typestring
				)

		return {
			'word':word,
			'is_relational': is_relational,
			'is_partial': is_partial,
			'subtype-code': subtype_code,
			'subtype': subtype
		}

