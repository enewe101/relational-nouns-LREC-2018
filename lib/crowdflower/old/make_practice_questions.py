from xml.dom import minidom

question_specs = [
		{
			'query':'supplier', 
			'correct':'role', 
			'binary-correct': 'relational',
			'ternary-correct': 'relational',
			'ternary-response': (
				'&ldquo;Supplier&rdquo; is used to establish a relationship '
				'between companies, where the supplier supplies parts or '
				'materials to the other company (the relatum).  Therefore it '
				'is relational.'
			),
			'response': (
				'Delegate is relational, because a delegate is defined in '
				'terms of that which it is appointed for (e.g. a delegate of '
				'the committee).  It is a social role.'
			)
		},
		{
			'query':'energy', 'correct':'property', 
			'binary-correct': 'relational',
			'ternary-correct': 'non-relational',
			'ternary-response': (
				'&ldquo;Energy&rdquo; does not indicate a relationship, so it '
				'is non-relational.'
			),
			'response': (
				'Energy is relational, because it can describe a property or '
				'characteristic.  E.g. &ldquo;He is energy is really '
				'positive&rdquo; '
				'or &ldquo;The energy of the reaction dwindled with time&rdquo;.'
			)
		},
		{
			'query':'rapport', 'correct':'connection',
			'binary-correct': 'relational',
			'ternary-correct': 'non-relational',
			'response': (
				'Although Rapport describes a connection between people, '
				'e.g. &ldquo;the rapport between them '
				'was very positive&rdquo;, it is not itself relational.'
			),
			'ternary-response': (
				'&ldquo;Rapport&rdquo; is a bridge-noun: '
				'&ldquo;the rapport between the students and teacher was '
				'breaking down.&rdquo;  Therefore it is not relational.'
			)
		},
		{
			'query':'captain', 
			'correct':'role', 
			'binary-correct': 'relational',
			'ternary-correct': 'partly-relational',
			'ternary-response': (
				'&ldquo;Captain&rdquo; can either refer to the rank, or to '
				'the leadership position in command of a ship. The former '
				'usage is non-relational, while the latter is relational. '
				"The relational usage doesn't seem distinctly more common, so "
				'we label &ldquo;captain&rdquo; partly relational.'
			)
		},
		{
			'query':'dispute', 'correct':'connection',
			'binary-correct': 'relational',
			'ternary-correct': 'non-relational',
			'ternary-response': (
				'&ldquo;Dipute&rdquo; is a bridge-noun: '
				'&ldquo;the dipute between business partners would not be '
				'resolved easily.&rdquo; Therefore it is non-relational.'
			),
			'response': (
				'Dispute is a connection: &ldquo;the dispute between the '
				'customer and service agent was to be expected&rdquo;.'
			)
		},
		{
			'query':'contract', 'correct':'connection',
			'binary-correct': 'relational',
			'ternary-correct': 'non-relational',
			'ternary-response': (
				'&ldquo;Contract&rdquo; is a bridge-noun: '
				'&ldquo;the contract between the parties is binding&rdquo;. '
				'Therefore it is non-relational.'
			),
			'response': (
				'Contract is a connection: &ldquo;the contract between both '
				'parties is binding&rdquo;.'
			)
		},
		{
			'query':'reporter', 
			'correct':'role', 
			'binary-correct': 'relational',
			'ternary-correct': 'partly-relational',
			'ternary-response': (
				'&ldquo;Reporter&rdquo; is a role, which can certainly be '
				'used non-relationally, as in : &ldquo;What do I do for a '
				"living? I'm a reporter.&rdquo; "
				'However, it is very often used to designate a persons '
				'affiliation to a particular news organization, as in: '
				'&ldquo;Neha Thirani Bagri is a reporter with The New York '
				"Times.&rdquo; It isn't clear which is more "
				'common, so we consider &ldquo;reporter&rdquo; to be '
				'partly relational.'
			)
		},
		{
			'query':'gender', 'correct':'property', 
			'binary-correct': 'relational',
			'ternary-correct': 'non-relational',
			'ternary-response': (
				"Gender is a property, so it is non-relational."
			),
			'response': (
				"Gender is a property of a person, so it's relational."
			)
		},
		{
			'query':'reconnaissance', 'correct':'non-relational', 
			'binary-correct': 'non-relational',
			'ternary-correct': 'non-relational',
			'ternary-response': (
				'&ldquo;Reconnaisance&rdquo; does not indicate a '
				'relationship.  Therefore it is non-relational.'
			),
			'response': (
				'Reconnaisance is not relational.  It describes the '
				"collection of knowledge, but isn't described in terms of "
				'a relationship to something else.'
			)
		},
		{
			'query':'tip', 'correct':'part', 
			'binary-correct': 'relational',
			'ternary-correct': 'relational',
			'ternary-response': (
				'Tip is relational, it describes the furthest point along '
				'some extended part of an object.  That means it is a '
				'relative part.'
			),
			'response': (
				'Tip is relational, it describes the furthest point along '
				'some extended part of an object.  That means it is a '
				'relative part.'
			)
		},
		{
			'query':'war', 'correct':'connection',
			'binary-correct': 'relational',
			'ternary-correct': 'non-relational',
			'ternary-response': (
				'&ldquo;War&rdquo; is a bridge-noun: '
				'&ldquo;The war between the '
				'countries raged on for decades&rdquo;. '
				'Therefore it is non-relational.'
			),
			'response': (
				'War is a connection: &ldquo;The war between the '
				'countries raged on for decades&rdquo;.'
			)
		},
		{
			'query':'operator', 
			'correct':'role', 
			'binary-correct': 'relational',
			'ternary-correct': 'non-relational',
			'ternary-response': (
				'&ldquo;Operator&rdquo; is a role, designating someone '
				'controlling machinery or a production process. As a role, '
				'it is conceivable that it could be used relationally, '
				'however that seems like a stretch, and such a usage is '
				'probably very rare.  Therefore, we consider '
				'&ldquo;operator&rdquo; to be '
				'non-relational.'
			)
		},
		{
			'query':'copper', 'correct':'non-relational', 
			'binary-correct': 'non-relational',
			'ternary-correct': 'non-relational',
			'ternary-response': (
				'&ldquo;Copper&rdquo; does not indicate a relationship.  '
				'Therefore it is '
				'non-relational.'
			),
			'response': (
				"Copper is non-relational -- it isn't defined as a "
				'relationship with something else.'
			)
		},
		{
			'query':'nail', 'correct':'part', 
			'binary-correct': 'relational',
			'ternary-correct': 'non-relational',
			'ternary-response': (
				'&ldquo;Nail&rdquo; does not indicate a relationship.  '
				'Therefore it is non-relational.'
			),
			'response': (
				'Nail is relational because it is a body part.'
			)
		},
		{
			'query':'realization', 'correct':'nominalized-verb', 
			'binary-correct': 'relational',
			'ternary-correct': 'non-relational',
			'ternary-response': (
				'&ldquo;Realization&rdquo; does not indicate a relationship.  '
				'Therefore it is non-relational.'
			),
			'response': (
				'Realization is relational.  It is a nominalized verb (to '
				'realize), and you can have a realization of something.'
			)
		},
		{
			'query':'writer', 
			'correct':'role', 
			'binary-correct': 'relational',
			'ternary-correct': 'partly-relational',
			'ternary-response': (
				'&ldquo;Writer&rdquo; has a few meanings: it can refer to the '
				'vocation, it can '
				'refer to the role at a newspaper company, or it can refer to '
				'the person who wrote the screenplay of a movie (although '
				'screenwriter isolates that meaning). The first non-relational '
				'usage is probably the most common. The latter two usages, as '
				'roles, can be used relationally, and it is not excessively '
				'rare to see them used that way. '
				'Therefore, we consider &ldquo;writer&rdquo; to be partly '
				'relational.'
			)
		},
		{
			'query':'noon', 'correct':'non-relational', 
			'binary-correct': 'non-relational',
			'ternary-correct': 'non-relational',
			'response': (
				'Noon is non-relational because it does not denote a '
				'relationship.'
			),
			'ternary-response': (
				'Noon is non-relational because it does not denote a '
				'relationship.'
			)

		},
		{
			'query':'head', 
			'correct':'role', 
			'binary-correct': 'relational',
			'ternary-correct': 'partly-relational',
			'ternary-response': (
				'&ldquo;Head&rdquo; has a few prominant meanings: it can '
				'signifiy the body part, it can signify a distal and bulbous '
				'part of an object, or it can signify a position of '
				'leadership.  The latter two meanings are relational, but the '
				'first meaning is probably more common.  We therefore label '
				'&ldquo;head&rdquo; as partly-relational.'
			)
		},
		{
			'query':'neck', 
			'correct':'role', 
			'binary-correct': 'relational',
			'ternary-correct': 'partly-relational',
			'ternary-response': (
				'&ldquo;Neck&rdquo; can mean the body part of an animal '
				'joining the head to the body, or it can mean an elongate '
				'constricted part of an object, such as a bottle or vase. '
				'The former usage is non-relational, while the latter is '
				'relational.  We think that the latter usage is less '
				'frequent, so we label &ldquo;neck&rdquo; as partly-relational.'
			)
		},
		{
			'query':'delegate', 'correct':'role', 
			'binary-correct': 'relational',
			'ternary-correct': 'relational',
			'ternary-responseresponse': (
				'&ldquo;Delegate&rdquo; is relational, because a delegate is '
				'defined in '
				'terms of that which it is appointed for (e.g. a delegate of '
				'the internal committee).'
			),
			'response': (
				'Delegate is relational, because a delegate is defined in '
				'terms of that which it is appointed for (e.g. a delegate of '
				'the committee).  It is a social role.'
			)
		},
		{
			'query':'invester', 
			'correct':'role', 
			'binary-correct': 'relational',
			'ternary-correct': 'partly-relational',
			'ternary-response': (
				'&ldquo;Invester&rdquo; is a role. It can be used to '
				'designate someone who frequently invests in companies and '
				'projects generally, or it can be used to signify the '
				'relationship between the person who invests and the '
				'company / project '
				'in which she invests&mdash;the latter being a relational '
				'usage.  We '
				'think that the relational usage is roughly on par with the '
				'generic usage, and since it is not clearly more common, we '
				'choose to label &ldquo;invester&rdquo; as partly-relational.'
			)
		},
		#{
		#	'query':'athlete', 'correct':'non-relational', 
		#	'binary-correct': 'non-relational',
		#	'ternary-correct': 'non-relational',
		#	'response': (
		#		'Athlete is non-relational.  It is not essentially defined as '
		#		'a relationship to something / someone else.'
		#	)
		#},
		#{
		#	'query':'mathematician', 'correct':'non-relational', 
		#	'binary-correct': 'non-relational',
		#	'ternary-correct': 'non-relational',
		#	'response': (
		#		'Mathematician is non-relational, it is not essentially '
		#		'defined as a relationship to something / someone else.'
		#	)
		#},
		#{
		#	'query':'opener', 'correct':'part', 
		#	'binary-correct': 'relational',
		#	'ternary-correct': 'partly-relational',
		#	'response': (
		#		'Opener can be used to refer to a tool, which is '
		#		'non-relational but it can also be used in a relational sense, '
		#		'where it describes the beginning part of a show '
		#		'e.g. &ldquo;the opener to her act set '
		#		'the expectations very high&rdquo;. Therefore we consider it '
		#		'partly relational.'
		#	)
		#},
		#{
		#	'query':'scientist', 'correct':'non-relational', 
		#	'binary-correct': 'non-relational',
		#	'ternary-correct': 'non-relational',
		#	'response': (
		#		'Scientist is non-relational.  It is not essentially defined '
		#		'as a relationship to something / someone else.'
		#	)
		#},
		#{
		#	'query':'predecessor', 'correct':'role', 
		#	'binary-correct': 'relational',
		#	'ternary-correct': 'relational',
		#	'response': (
		#		'Predecessor is inherently relational, being defined as the '
		#		'thing which came before something else.  This makes it a '
		#		'non-social role.'
		#	)
		#},
		#{
		#	'query':'mayor', 'correct':'role', 
		#	'binary-correct': 'relational',
		#	'ternary-correct': 'relational',
		#	'response': (
		#		'Mayor is relational, being defined as being in charge of a '
		#		'city.  This makes it a social role.'
		#	)
		#},
		#{
		#	'query':'blacksmith', 'correct':'non-relational', 
		#	'binary-correct': 'non-relational',
		#	'ternary-correct': 'non-relational',
		#	'response': (
		#		'Blacksmith is non-relational.  Although it is a vocation, '
		#		"it isn't essentially defined in terms of a relationship to "
		#		'something else.'
		#	)
		#},
		#{
		#	'query':'partner', 'correct':'role', 
		#	'binary-correct': 'relational',
		#	'ternary-correct': 'relational',
		#	'response': (
		#		'Partner is inherently relational, describing a person who '
		#		"is cooperating with someone else.  That means it's a social "
		#		'role.'
		#	)
		#},
		#{
		#	'query':'assistant', 'correct':'role', 
		#	'binary-correct': 'relational', 
		#	'ternary-correct': 'relational', 
		#	'response': (
		#		'Assistant is inherently relational, describing a person who '
		#		"is supporting someone else.  That means it's a social role."
		#	)
		#},
]

BINARY_OPTION_TYPES = ['non-relational', 'relational']
TERNARY_OPTION_TYPES = ['non-relational', 'partly-relational', 'relational']
OPTION_TYPES = [
	'non-relational', 'role', 'connection', 'part', 'property', 'verblike'
]

DOM = minidom.Document()

def element(tag_name, attributes={}):
    elm = DOM.createElement(tag_name)
    bind_attributes(elm, attributes)
    return elm 

def bind_attributes(element, attributes):
    for attribute in attributes:
        element.setAttribute(attribute, attributes[attribute])
    return element

def div(attributes={}):
    return element('div', attributes)

def span(attributes={}):
    return element('span', attributes)

def text(text_content):
    return DOM.createTextNode(text_content)


def make_quiz_questions(arity='ternary'):
	questions_container = div()
	for i, spec in enumerate(question_specs):
		questions_container.appendChild(make_quiz_question(i, spec, arity))

	return questions_container.toprettyxml()

def make_quiz_question(i, spec, arity='ternary'):
	question_wrapper = span({'class':'quiz'})

	# Make query part
	query_line = question_wrapper.appendChild(span({'class':'queryline'}))
	query = query_line.appendChild(span({'class': 'query'}))
	query_word = query.appendChild(span({'class':'query-word'}))
	query_word.appendChild(text(spec['query']))

	# Answer
	correct = query_line.appendChild(span({'class':'correct-answer'}))
	correct.appendChild(
		text(spec['ternary-correct']) if arity == 'ternary'
		else text(spec['binary-correct']) if arity == 'binary'
		else text(spec['correct'])
	)

	# Make and append options
	option_wrapper = query_line.appendChild(div({'class':'option-wrapper'}))
	for option in make_options(i, arity):
		option_wrapper.appendChild(option)

	# Make response portion
	response_line = question_wrapper.appendChild(span({'class':'responseline'}))
	response = response_line.appendChild(span({'class':'response'}))
	prefix = response.appendChild(span({'class':'prefix'}))
	prefix.appendChild(text('prefix'))
	use_response = (
		spec['response'] if arity != 'ternary'
		else spec['ternary-response'] if 'ternary-response' in spec
		else spec['response']
	)
	response.appendChild(text(use_response))

	return question_wrapper


def make_options(i, arity='ternary'):
	options = []
	option_types = (
		TERNARY_OPTION_TYPES if arity == 'ternary'
		else BINARY_OPTION_TYPES if arity == 'binary'
		else OPTION_TYPES
	)

	for j, option in enumerate(option_types):
		option_elm = span({'class':option+'-option option'})
		option_elm.appendChild(element(
			'input', 
			{'type':'radio', 'id':'%s.%s'%(i,j), 'name':'%s.'%i}
		))
		label = option_elm.appendChild(element('label', {'for':'%s.%s'%(i,j)}))
		label.appendChild(text(option))
		options.append(option_elm)

	return options

