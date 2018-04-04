from xml.dom import minidom
import random
import csv

cf_test_questions_0 = [

	# From basic examples
	'agreement', 'aunt', 'box', 'bread', 'connection', 'direction', 'fever',
	'flower', 'husband', 'mentor', 'mother', 'nephew', 'sibling',

	# From functional relationships
	'author', 'building', 'legacy', 'passenger', 'province',
	'subsidiary',
	# 'hole', 

	# From relative parts
	'clip', 'corner', #'crust', 
	'handle', 'middle', 'part', 'shelf', 'strap',

	# From roles
	'CEO', 'babysitter', 'director', 'miner',
	#'lawyer', 

	# From self test
	'assistant', 'athlete', 'blacksmith', 'copper', 'delegate', 'foreigner',
	'head', 'manufacturer', 'nail', 'noon', 'operator', 'pair', 'partner',
	'producer', 'rapport', 'realization', 'reconnaissance', 'reporter',
	'scientist', 'stranger', 'supplier', 'top', 'war',
	#'discovery', 'captain', 'creation', 'investor', 'loser', 'tip', 
]

cf_test_questions_1 = [

	# Aditional positives
	'colleague', 'client', 'coach', 'stepmother', 'nominee', 'viewer',
	'parent', 'admiral', 'estate', 'back', 'ammendment', 
	'alias',
	'teammate', 'playmate', 'nemesis', 'namesake', 'founder', 'manager',
	'contributor', 'kin', 'adviser', 'minder', 'consultant', 'primier',
	'seatmate',  'painter', 'contractor', 'associate',

	# Additional negatives
	'telephone', 'program', 'flour', 'norm', 'vehicle',
	'oversight', 'kiss', 'signing', 'oven', 'poker', 'rock', 'purse',
	'reverence', 'protestation', 'inhibition', 'betrayal', 'orchestration',
	'turn', 'game', 'consideration', 'malpractice', 'kick', 'statement',
	'furniture'
]



question_groupings = {
	'basic_examples': [
		'friendship', 'bus', 'entourage', 'friend', 'confidante', 'coworker', 
		'book', 'relative', 'mannequin', 'conflict'
	],

	'functional_relationships': [
		'roof', 'delivery', 'solution', 'replacement', 'adherence', 'successor', 
		'heir', 'girder', 'organization', 'distribution'
	],

	'relative_parts': [
		'cord', 'door', 'edge', 'front', 'outside', 'rail', 'stern', 'stirrup',
		'screw', 'wire'
	],

	'roles': [ 
		'ambassador', 'astronaut', 'gardener', 'guitarist', 'pitcher', 'planner',
		'president', 'purchaser', 'shareholder', 'supervisor'
	],

	'self_test': [
		'base', 'contract', 'dispute', 'energy', 'gender', 'mathematician', 'mayor',
		'neck', 'predecessor', 'writer'
	]
}


question_specs = {
	'mother': {
		'query': 'mother',
		'correct': 'relational',
		'response': (
			'&ldquo;Mother&rdquo; expresses the mother-child '
			'relationship and refers to one of the relata (the mother), '
			'therefore it is '
			'relational'
			'.'
		)
	},
	'aunt': {
		'query': 'aunt',
		'correct': 'relational',
		'response': (
			'&ldquo;Aunt&rdquo; expresses the aunt-nephew or aunt-niece '
			'relationship and refers to one of the relata (the aunt), '
			'therefore it is '
			'relational'
			'.'
		)
	},
	'husband': {
		'query': 'husband',
		'correct': 'relational',
		'response': (
			'&ldquo;Husband&rdquo; expresses a spousal '
			'relationship, and refers to one of the relata, '
			'therefore it is '
			'relational'
			'.'
		)
	},
	'nephew': {
		'query': 'nephew',
		'correct': 'relational',
		'response': (
			'&ldquo;Nephew&rdquo; expresses a aunt-nephew or uncle-nephew '
			'relationship, and refers to one of the relata, '
			'therefore it is '
			'relational'
			'.'
		)
	},
	'relative': {
		'query': 'relative',
		'correct': 'relational',
		'response': (
			'&ldquo;Relative&rdquo; expresses a generic blood-relative '
			'relationship, and refers to one of the relata, '
			'therefore it is '
			'relational'
			'.'
		)
	},
	'friend': {
		'query': 'friend',
		'correct': 'relational',
		'response': (
			'&ldquo;Friend&rdquo; expresses the friendship relationship '
			'and refers to one of the relata, '
			'therefore it is '
			'relational'
			'.'
		)
	},
	'mentor': {
		'query': 'mentor',
		'correct': 'relational',
		'response': (
			'&ldquo;Mentor&rdquo; expresses the mentor-mentee relationship '
			'and refers to one of the relata, '
			'therefore it is '
			'relational'
			'.'
		)
	},
	'ally': {
		'query': 'ally',
		'correct': 'relational',
		'response': (
			'&ldquo;Ally&rdquo; expresses an alliance relationship, and '
			'refers to one of the relata, '
			'therefore it is '
			'relational'
			'.'
		)
	},
	'coworker': {
		'query': 'coworker',
		'correct': 'relational',
		'response': (
			'&ldquo;Coworker&rdquo; expresses the relationship between '
			'peers working in the same place, and refers to one of the '
			'relata, therefore it is '
			'relational'
			'.'
		)
	},
	'entourage': {
		'query': 'entourage',
		'correct': 'relational',
		'response': (
			'&ldquo;Entourage&rdquo; expresses a relationship between '
			'a person (usually a celebrity) and their friends and '
			'assistants who travel with them.  It refers to one of the '
			'relata (the friends and assistants as a group), '
			'so it is '
			'relational'
			'.'
		)
	},
	'confidante': {
		'query': 'confidante',
		'correct': 'relational',
		'response': (
			'&ldquo;Confidante&rdquo; expresses a relationship between '
			'one person and another with whom they share personal or '
			'sensitive information.  Also, it refers to one of the relata '
			'(the recipient of the information), therefore it is '
			'relational'
			'.'
		)
	},
	'sibling': {
		'query': 'sibling',
		'correct': 'relational',
		'response': (
			'&ldquo;Sibling&rdquo; expresses the siblinghood relationship '
			'between brothers and / or sisters, and when used it refers to '
			'one of the relata, '
			'therefore it is '
			'relational'
			'.'
		)
	},
	'bread': {
		'query': 'bread',
		'correct': 'non-relational',
		'response': (
			"&ldquo;Bread&rdquo; does not express a relationship, "
			'so it is '
			'non-relational'
			'.'
		)
	},
	'fever': {
		'query': 'fever',
		'correct': 'non-relational',
		'response': (
			"&ldquo;Fever&rdquo; does not express a relationship, "
			'so it is '
			'non-relational'
			'.'
		)
	},
	'book': {
		'query': 'book',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Book&rdquo; does not express a relationship, '
			'so it is '
			'non-relational'
			'.'
		)
	},
	'box': {
		'query': 'box',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Box&rdquo; does not express a relationship, '
			'so it is '
			'non-relational'
			'.'
		)
	},
	'flower': {
		'query': 'flower',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Flower&rdquo; does not express a relationship, '
			'so it is '
			'non-relational'
			'.'
		)
	},
	'bus': {
		'query': 'bus',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Bus&rdquo; does not express a relationship, '
			'so it is '
			'non-relational'
			'.'
		)
	},
	'mannequin': {
		'query': 'mannequin',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Mannequin&rdquo; does not express a relationship, '
			'so it is '
			'non-relational'
			'.'
		)
	},
	'direction': {
		'query': 'direction',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Direction&rdquo; can be used in a few different ways, but '
			'none of them are relational.  It can be used as a property as in '
			'&ldquo;the direction of the wind&rdquo;.  Or it can be used to '
			'mean essentially &ldquo;instrucitons&rdquo; as in &ldquo;The '
			'cooking directions&rdquo;  It can also mean specifically '
			'navigation instructions too.  But none of these uses are '
			'relational.'
		)
	},
	'friendship': {
		'query': 'friendship',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Friendship&rdquo; does express a relationship, '
			'but it does not refer to one of the relata.  It refers to '
			"the relationship itself, so it is "
			'non-relational'
			'.'
		)
	},
	'conflict': {
		'query': 'conflict',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Conflict&rdquo; does express a relationship, but it '
			"doesn't refer to one of the relata of that relationship. "
			"It refers to the relationship itself, so it is "
			'non-relational'
			'.'
		)
	},
	'alliance': {
		'query': 'alliance',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Alliance&rdquo; does express a relationship, but it '
			"doesn't refer to one of the relata of that relationship. "
			"It refers to the relationship itself, so it is "
			'non-relational'
			'.'
		)
	},
	'connection': {
		'query': 'connection',
		'correct': 'non-relational',
		'response': (
			"&ldquo;Connection&rdquo; does express a relationship, but "
			"it doesn't refer to one of the relata of the relationship. "
			"It refers to the relationship itself, so it is "
			'non-relational. This is similar to &ldquo;contract&rdquo;.'
		)
	},
	'agreement': {
		'query': 'agreement',
		'correct': 'non-relational',
		'response': (
			"&ldquo;Agreement&rdquo; does express a relationship, but "
			"it doesn't refer to one of the relata of that relationship. "
			"It refers to the relationship itself, so it is "
			'non-relational.  This is similar to &ldquo;contract&rdquo;'
			'.'
		)
	},
	'passenger': {
		'query': 'passenger',
		'correct': 'relational',
		'response': (
			'&ldquo;Passenger&rdquo; is inherently defined in terms of '
			'a relationship between a person and a mode of transport. '
			'Since it refers to one of the relata of that '
			"relationship, it's a relational noun.  Even in cases like "
			'&ldquo;Passengers could face delays&rdquo;, it is still '
			'establishing a relationship to a mode of transit '
			"even if that mode isn't mentioned, i.e. the previous "
			'example is short for &ldquo;Passengers of American Airlines '
			'could face delays.  Therefore we consider passenger to '
			'be relational.'
		)
	},
	'author': {
		'query': 'author',
		'correct': ['relational','partly-relational'],
		'response': (
			'&ldquo;Author&rdquo; is typically used to establish a '
			'relationship between a written work and its creator.  Since '
			"it refers to one of the relata (the creator) it's a "
			'relational.  It can also be used non-relationally, to describe the '
			'vocation, as in &rdquo;She is an author&ldquo;, so we also '
			'accept occasionally relational as a correct response.' 
		)
	},
	'subsidiary': {
		'query': 'subsidiary',
		'correct': 'relational',
		'response': (
			'&ldquo;Subsidiary&rdquo; expresses the relationship between '
			'a parent company and another company owned by the parent. '
			'It refers to one of the relata (the owned company), so it is '
			'a '
			'relational'
			' noun.'
		)
	},
	'replacement': {
		'query': 'replacement',
		'correct': 'relational',
		'response': (
			'&ldquo;Replacement&rdquo; expresses a relationship between '
			'an original object and a secondary object that '
			'substitutes the original.  It refers to one of the relata '
			"(the substitute) so it's a "
			'relational'
			' noun.'
		)
	},
	'successor': {
		'query': 'successor',
		'correct': 'relational',
		'response': (
			'&ldquo;Successor&rdquo; expresses a relationship '
			'between one object and second that follows '
			'in some kind of lineage.  It refers to one of '
			'the relata (the secondary object), so it is a '
			'relational'
			' noun.'
		)
	},
	'hole': {
		'query': 'hole',
		'correct': 'relational',
		'response': (
			'&ldquo;Hole&rdquo; expresses the relationship between some '
			'self-connected entity and an enclosed gap in that entity. '
			'It refers to one of the relata (the gap) so it is a '
			'relational'
			' noun.'
		)
	},
	'solution': {
		'query': 'solution',
		'correct': 'relational',
		'response': (
			'&ldquo;Solution&rdquo; expresses the relationship between '
			'some problem and the agent or technique that solves it. '
			"It refers to one of the relata, so it's a "
			'relational'
			' noun.'
		)
	},
	'province': {
		'query': 'province',
		'correct': 'relational',
		'response': (
			'&ldquo;Province&rdquo; expresses a relationship between a '
			'one geo-political entity and another that subsumes it. '
			'It refers to one of the '
			"relata (the subsumed geo-political entity) so it's "
			'a relational '
			'noun.  Even when the nation is not mentioned, it is implied. '
			'For example, saying &ldquo;Quebec is a great province&rdquo;, '
			'is short for saying &ldquo;Quebec is a great province of '
			'Canada.&rdquo;'
		)
	},
	'heir': {
		'query': 'heir',
		'correct': 'relational',
		'response': (
			'&ldquo;Heir&rdquo; expresses the relationship between a '
			'person and their inheritance.  It refers to one of those '
			'relata (the person), so it is a '
			'relational'
			' noun.'
		)
	},
	'legacy': {
		'query': 'legacy',
		'correct': 'relational',
		'response': (
			'&ldquo;Legacy&rdquo; expresses a relationship between a '
			'person and the set of notable things that they will be '
			'remembered for.  It refers to one of the relata (the person) '
			"so it's a relational noun."
		)
	},
	'distribution': {
		'query': 'distribution',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Distribution&rdquo; does not express a relationship. '
			'It expresses either the act of distributing something, or '
			'the manner in which it has been distributed.  Since it '
			"doesn't express a relationship, it isn't a "
			'relational'
			' noun.'
		)
	},
	'organization': {
		'query': 'organization',
		'correct': 'non-relational',
		'response': (
			"&ldquo;Organization&rdquo; doesn't express a relationship. "
			'It expresses the act of organizing, the manner in which '
			'something is organized, or it expresses an administrative '
			"body like a company or institution.  Since it doesn't "
			"express a relationship, it is "
			'non-relational'
			'.'
		)
	},
	'delivery': {
		'query': 'delivery',
		'correct': 'non-relational',
		'response': (
			"&ldquo;Delivery&rdquo; doesn't express a "
			'relationship, but rather expresses either the act of '
			'delivering, '
			'as in "She went home after she finished her delivery", or '
			'a performance (usually a show or presentation), '
			'as in "Her delivery was smooth and well-practiced". '
			"Since delivery does not express a relationship, it is "
			'non-relational'
			'.'
		)
	},
	'girder': {
		'query': 'girder',
		'correct': 'non-relational',
		'response': (
			"&ldquo;Girder&rdquo; doesn't express a relationship so "
			"it is "
			'non-relational'
			'.'
		)
	},
	'roof': {
		'query': 'roof',
		'correct': 'non-relational',
		'response': (
			"&ldquo;Roof&rdquo; doesn't express a relationship so "
			"it is "
			'non-relational'
			'.'
		)
	},
	'building': {
		'query': 'building',
		'correct': 'non-relational',
		'response': (
			"&ldquo;Building&rdquo; either refers to a permanent "
			'construction made to hold people and physical objects, or '
			'the act of constructing something.  Neither meaning '
			'expresses a relationship, so &ldquo;building&rdquo; is '
			'non-relational'
			'.'
		)
	},
	'adherence': {
		'query': 'adherence',
		'correct': 'non-relational',
		'response': (
			'While &ldquo;adherence&rdquo; can, in a sense, be used to '
			'express a '
			'relationship (as in "the adherence of the stickers to the '
			'page") it '
			"doesn't refer to one of the relata, so it is "
			'non-relational'
			'.'
		)
	},
	'corner': {
		'query': 'corner',
		'correct': 'relational',
		'response': (
			'&ldquo;Corner&rdquo; identifies a part of a physical object '
			'where edges or faces meet to form a sharp protrusion '
			'It establishes a physical / geometric relationship between '
			'the part to which it refers (a relatum) and the whole object '
			'so it is a '
			'relational'
			' noun.'
		)
	},
	'middle': {
		'query': 'middle',
		'correct': 'relational',
		'response': (
			'&ldquo;Middle&rdquo; identifies a part of an object extended '
			"in space or time based on it's spatial or temporal "
			'relationship to the whole.  E.g. &ldquo;The middle of the '
			'planet&rdquo; or &ldquo;The middle of the meeting&rdquo; '
			'Since it establishes a '
			"relationship while referring to one of the relata, it's a "
			'relational'
			' noun.'
		)
	},
	'part': {
		'query': 'part',
		'correct': 'relational',
		'response': (
			'&ldquo;Part&rdquo; expresses the whole-part relationship, and '
			'refers to one of the relata in that relationship (the part), '
			"so it's a "
			'relational'
			' noun.'
		)
	},
	'base': {
		'query': 'base',
		'correct': 'occasionally relational',
		'response': (
			'&ldquo;Base&rdquo; can be a relative part, signifying the '
			'bottom part of an object, '
			'but it can also refer to an outpost where operatives '
			'are located.  We consider '
			'&ldquo;base&rdquo; to be '
			'occasionally relational.'
		)
	},
	'top': {
		'query': 'top',
		'correct': ['relational', 'partly-relational'],
		'response': (
			'&ldquo;Top&rdquo; can express a physical relationship between '
			'a whole object and its uppermost part.  It can also refer to '
			'a toy that played with by spinning it.  We think that the '
			'latter meaning, which satisfies the relational noun criteria '
			'is the most common meaning, so we consider &ldquo;top&rdquo; '
			'to be a relational noun.  However, we also accept occasionally '
			'relational'
		)
	},
	'edge': {
		'query': 'edge',
		'correct': 'relational',
		'response': (
			'&ldquo;Edge&rdquo; identifies part of an object in terms of '
			'its physical relationship to the whole, so it is a '
			'relational'
			' noun.'
		)
	},
	'outside': {
		'query': 'outside',
		'correct': 'relational',
		'response': (
			'&ldquo;Outside&rdquo; identifies a region based on its '
			'relationship to some object, so it is a '
			'relational'
			' noun.'
		)
	},
	'stern': {
		'query': 'stern',
		'correct': 'relational',
		'response': (
			'&ldquo;Stern&rdquo; identifies a part of a ship based on '
			"its spatial relationship to the rest of the ship, so it's a "
			'relational'
			' noun.'
		)
	},
	'front': {
		'query': 'front',
		'correct': 'relational',
		'response': (
			'&ldquo;Front&rdquo; identifies part of an object based on '
			'its physical relationship to the whole, so it is a '
			'relational'
			' noun.'
		)
	},
	'screw': {
		'query': 'screw',
		'correct': 'non-relational',
		'response': (
			'Although a &ldquo;screw&rdquo; is usually part of another '
			"object, the meaning of screw doesn't specifically denote "
			'such a relationship, so &ldquo;screw&rdquo; is '
			'non-relational'
			'.'
		)
	},
	'wheel': {
		'query': 'wheel',
		'correct': 'non-relational',
		'response': (
			'Although a &ldquo;wheel&rdquo; is usually part of another '
			"object, the meaning of wheel doesn't specifically denote "
			'such a relationship, so &ldquo;wheel&rdquo; is '
			'non-relational'
			'.'
		)
	},
	'strap': {
		'query': 'strap',
		'correct': 'non-relational',
		'response': (
			'Although a &ldquo;strap&rdquo; is usually part of another '
			"object, the meaning of strap doesn't specifically denote "
			'such a relationship, so &ldquo;strap&rdquo; is '
			'non-relational'
			'.'
		)
	},
	'door': {
		'query': 'door',
		'correct': 'non-relational',
		'response': (
			'Although a &ldquo;door&rdquo; is usually part of another '
			"object, the meaning of &ldquo;door&rdquo; doesn't "
			'specifically denote '
			'such a relationship, so it is '
			'non-relational'
			'.'
		)
	},
	'shelf': {
		'query': 'shelf',
		'correct': 'non-relational',
		'response': (
			'Although a &ldquo;shelf&rdquo; can a be part of another '
			"object, the meaning of &ldquo;shelf&rdquo; doesn't "
			'specifically denote '
			'such a relationship, so it is '
			'non-relational'
			'.'
		)
	},
	'handle': {
		'query': 'handle',
		'correct': 'non-relational',
		'response': (
			'Handle can mean a part of an object meant to be graped in the '
			'hand.  It can also mean the name used to address another '
			"user on social media such as Twitter.  We don't think that "
			"the usage to refer to a user is dominant (yet) so we take handle "
			'to only be occasionally relational.'
		)
	},
	'cord': {
		'query': 'cord',
		'correct': 'non-relational',
		'response': (
			'Although a &ldquo;cord&rdquo; is typically part of another '
			"object, the meaning of &ldquo;cord&rdquo; doesn't "
			'specifically denote '
			'such a relationship, so it is '
			'non-relational'
			'.'
		)
	},
	'clip': {
		'query': 'clip',
		'correct': ['relational', 'partly-relational'],
		'response': (
			'&ldquo;Clip&rdquo; can refer to a small clamp or fastening, '
			'or it can refer to a part of a recording. '
			'The first usage is non-relational and the second is relational. '
			'We think the second usage accounts for more than half of cases, '
			'so we consider &ldquo;clip&rdquo; to be relational. '
			'However, we also accept occasionally relational as a correct '
			'respoonse.'
		)
	},
	'stirrup': {
		'query': 'stirrup',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Stirrup&rdquo; does not denote a relationship, so '
			"it is "
			'non-relational'
			'.'
		)
	},
	'wire': {
		'query': 'wire',
		'correct': 'non-relational',
		'response': (
			'Although a &ldquo;wire&rdquo; is typically part of another '
			"object, the meaning of &ldquo;wire&rdquo; doesn't "
			'specifically denote '
			'such a relationship, so it is '
			'non-relational'
			'.'
		)
	},
	'rail': {
		'query': 'rail',
		'correct': 'non-relational',
		'response': (
			'Although a &ldquo;rail&rdquo; can part of another '
			"object, the meaning of &ldquo;rail&rdquo; doesn't "
			'specifically denote '
			'such a relationship, so it is '
			'non-relational'
			'.'
		)
	},
	'crust': {
		'query': 'crust',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Crust&rdquo; identifies part of an object (usually '
			'a baked good that is firmer, crunchier, and/or more cooked '
			'One could make the argument that the physical relationship '
			'of a crust to the rest of the object is the salient aspect '
			"of the word's meaning, and that it is therefore a "
			'relational noun.  We consider it simply to be defined in '
			'terms '
			'of its properties, and that the fact that it is typically '
			'located at the outer surface or edge of baked food to be '
			'a typical property of crust but not the essential meaning, '
			'so we consider &ldquo;crust&rdquo; to be '
			'non-relational'
			'. Nevertheless, a good argument could be made to consider '
			'it relational.'
		)
	},
	'director': {
		'query': 'director',
		'correct': 'relational',
		'response': (
			'&ldquo;Director&rdquo; identifies a role, and is usually '
			'used to indicate a leadership relationship to an '
			'organizational unit, as in "Director of Finance". '
			'It can also be used in the sense of "the director of the film", '
			'which is also a relational usage. '
			'Therefore, we judge '
			'&ldquo;director&rdquo; to be a '
			'relational'
			' noun.'
		)
	},
	'CEO': {
		'query': 'CEO',
		'correct': 'relational',
		'response': (
			'&ldquo;CEO&rdquo; identifies a role, and is usually '
			'used to indicate a leadership relationship to a '
			'company, as in "The CEO of Walmart".  '
			'But, CEO can also be used in a generic non-relational sense, '
			'as in &ldquo;CEO salaries have skyrocketed.&rdquo; '
			'We think '
			'the relational usage is more common, and judge '
			'&ldquo;CEO&rdquo; to be a '
			'relational'
			' noun.'
		)
	},
	'president': {
		'query': 'president',
		'correct': 'relational',
		'response': (
			'&ldquo;President&rdquo; identifies a role, and is usually '
			'used to indicate a leadership relationship to country '
			'as in "President of Italy".  We think '
			'the relational usage is most common, and judge '
			'&ldquo;president&rdquo; to be a '
			'relational'
			' noun.'
		)
	},
	'ambassador': {
		'query': 'ambassador',
		'correct': 'relational',
		'response': (
			'&ldquo;Ambassador&rdquo; identifies a role, and is usually '
			'used to indicate a relationship between that person and '
			'either their country of nationality, and / or the foreign '
			'country in which they are posted, '
			'as in "US Ambassador to France".  We think '
			'the relational usage is most common, and judge '
			'&ldquo;ambassador&rdquo; to be a '
			'relational'
			' noun.'
		)
	},
	'supervisor': {
		'query': 'supervisor',
		'correct': 'relational',
		'response': (
			'&ldquo;Supervisor&rdquo; identifies a role, and is usually '
			'used to indicate a leadership and oversight relationship to '
			'another person or group of people, '
			'as in "John\'s supervisor never lets him take breaks." '
			'We think the relational usage is most common, and judge '
			'&ldquo;supervisor&rdquo; to be a '
			'relational'
			' noun.'
		)
	},
	'pitcher': {
		'query': 'pitcher',
		'correct': 'relational',
		'response': (
			'&ldquo;Pitcher&rdquo; identifies a role on a baseball team, '
			'as in "Pitcher for the Mets".  We think '
			'the relational usage is most common, and judge '
			'&ldquo;pitcher&rdquo; to be a '
			'relational'
			' noun.'
		)
	},
	'guitarist': {
		'query': 'guitarist',
		'correct': 'partly-relational',
		'response': (
			'&ldquo;Guitarist&rdquo; can identify a role in a band, '
			'as in "Jimi Hendrix started as a guitarist for the Isley '
			'Brothers".  However, its non-relational usage, as in '
			'"Jimi Hendrix is a legendary guitarist" is also probably '
			'very common, and we judge &ldquo;guitarist&rdquo; to be '
			'occasionally relational'
			'.'
		)
	},
	'shareholder': {
		'query': 'shareholder',
		'correct': 'partly-relational',
		'response': (
			'&ldquo;Shareholder&rdquo; expresses the relationship '
			'between the owner of shares and the company or shares that '
			'are owned. But the word is also frequently used in a '
			'non-relational sense, meaning generically one who owns '
			'shares, without reference to the particular shares or '
			"company.  We don't think the relational usage is necessarily "
			'the most common, so we judge &ldquo;shareholder&rdquo; to be '
			'occasionally relational'
			'.'
		)
	},
	'lawyer': {
		'query': 'lawyer',
		'correct': 'partly-relational',
		'response': (
			'&ldquo;Lawyer&rdquo; can refer to a particular vocation. '
			'But it is also often used to indicate the relationship '
			'between the lawyer and the person (s)he is representing, '
			'as in "Julian Assange\'s lawyer was not available for '
			'comment." '
			'We don\'t think the relational usage is sufficiently common '
			'to consider &ldquo;lawyer&rdquo; a relational noun, so we '
			'judge it to be '
			'occasionally relational'
			'.'
		)
	},
	'babysitter': {
		'query': 'babysitter',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Babysitter&rdquo; identifies a person that watches '
			'over children.  While it could be used relationally, '
			'establishing a relationship to the children, we think that '
			'its core of the meaning is about the performance of duty '
			'rather than the relationship it implies with the children. '
			'Therefore we judge &ldquo;babysitter&rdquo; to be '
			'non-relational'
			', although an argument could be made otherwise.'
		)
	},
	'planner': {
		'query': 'planner',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Planner&rdquo; identifies a person who determines '
			'a sequence of actions or allocation of resources to achieve '
			'some goal.  The meaning implies the undertaking of an '
			"activity, but doesn't centrally express a relationship. "
			'Therefore we consider &ldquo;planner&rdquo; to be a '
			'non-relational'
			' noun.'
		)
	},
	'astronaut': {
		'query': 'astronaut',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Astronaut&rdquo; identifies a particular vocation, '
			'but does not centrally express a relationship.  Therefore '
			'we consider &ldquo;astronaut&rdquo; to be a '
			'non-relational'
			' noun.'
		)
	},
	'purchaser': {
		'query': 'purchaser',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Purchaser&rdquo; identifies a particular vocation, '
			'or generally a person responsible for the act of purchasing '
			'but does not centrally express a relationship.  Therefore '
			'we consider &ldquo;purchaser&rdquo; to be a '
			'non-relational'
			' noun.'
		)
	},
	'miner': {
		'query': 'miner',
		'correct': 'non-relational',
		'response': (
			'&ldquo;miner&rdquo; identifies a particular vocation, '
			'but does not centrally express a relationship.  Therefore '
			'we consider &ldquo;miner&rdquo; to be a '
			'non-relational'
			' noun.'
		)
	},
	'gardener': {
		'query': 'gardener',
		'correct': 'non-relational',
		'response': (
			'&ldquo;gardener&rdquo; identifies a particular vocation, '
			'but does not centrally express a relationship.  Therefore '
			'we consider &ldquo;gardener&rdquo; to be a '
			'non-relational'
			' noun.'
		)
	},
	'foreigner': {
		'query': 'foreigner',
		'correct': 'relational',
		'response': (
			'&ldquo;Foreigner&rdquo; expresses a relationship between '
			'a person and a country, wherein the person does not normally '
			'live in the country and was not born in the country. '
			'Since it expresses a relationship, and refers to one of the '
			'relata (the person), &ldquo;foreigner&rdquo; is a '
			'relational noun. Foreigner is a bit more tricky, because it '
			"isn't actually necessary to mention what country the person "
			'is foreign to. For example, one can say &ldquo;He is a '
			'foreigner&rdquo; and the country is generally understood to '
			'be the country that the speaker is in.'
		)
	},
	'stranger': {
		'query': 'stranger',
		'correct': 'relational',
		'response': (
			'&ldquo;Stranger&rdquo; expresses a relationship between '
			'people who are not familiar to each other, and when used it '
			'refers to one of those people (one of the relata). '
			'Therefore &ldquo;stranger&rdquo; is a '
			'relational noun.  Stranger is a bit more tricky, because often '
			"the other person isn't mentioned: for example, one often says "
			'&ldquo;She is a stranger&rdquo; rather than &ldquo;She is a '
			"stranger to me&rdquo;, but even when the other person isn't "
			'mentioned, they are implied.'
		)
	},
	'manufacturer': {
		'query': 'manufacturer',
		'correct': ['relational', 'partly-relational'],
		'response': (
			'&ldquo;Manufacturer&rdquo; expresses the relationship '
			'between a product and the entity (usually a company) that '
			'makes it.  While that meaning is relational, '
			'&ldquo;manufacturer&rdquo; can also be used in a generic '
			'non-relational sense, as in '
			'&ldquo;Manufacturers and exporters will be negatively '
			'affected this quarter&rdquo;.  We think the relational usage '
			'is most common, and judge &ldquo;manufacturer&rdquo; to be '
			'a relational noun.  However, we also accept occasionally '
			'relational.'
		)
	},
	'producer': {
		'query': 'producer',
		'correct': ['partly-relational', 'relational'],
		'response': (
			'&ldquo;Producer&rdquo; expresses the relationship '
			'between a product and the entity (usually a company) that '
			'produces it.  It can also describe the role in the creation '
			'of a film.  Both usages are relational.  There is also '
			'a generic non-relational usage, as in &ldquo;Producers and '
			'consumers are co-dependent in the economy,&rdquo; we think '
			'the relational usages are most commonly used, and judge '
			'&ldquo;producer&rdquo; to be a '
			'relational noun, but we also accept occasionally relational.'
		)
	},
	'discovery': {
		'query': 'discovery',
		'correct': 'partly-relational',
		'response': (
			'&ldquo;Discovery&rdquo; can refer to the act of '
			'discovering, as in &ldquo;The discover of Arsenic&rdquo;, '
			'or it can refer to the thing discovered, as in '
			'&ldquo;The law of the photoelectric effect is one of '
			'Einstein&rsquo;s important discoveries.&rdquo;. '
			'As shown, when used to refer to the thing discovered, it can '
			'be used to express the relationship between the person '
			'who did the discovering and the thing discovered. '
			'However, we think that the relational usage is not very '
			'common, and '
			'so judge &ldquo;discovery&rdquo; to be '
			'occasionally relational'
			'.'
		)
	},
	'creation': {
		'query': 'creation',
		'correct': ['partly-relational', 'relational'],
		'response': (
			'&ldquo;Creation&rdquo; can refer to the act of creating, or '
			'to the thing created, as in '
			'&ldquo;According to the Bible,&rsquo; the Universe is '
			'God&rsquo;s creation.&rdquo;  As shown, when it refers '
			'to the thing created, it can be used to establish the '
			'relationship between the creator and the thing created. '
			"We don't think the relational usage is dominant, so we "
			'judge creation to be only '
			'occasionally relational, but we still accpet the "usually '
			'relational" label.'
		)
	},
	'loser': {
		'query': 'loser',
		'correct': 'relational',
		'response': (
			'&ldquo;Loser&rdquo; can be used relationally to designate an '
			'entrant in '
			'a competition that lost, as in: '
			'&ldquo;The losers were given a certificate for their '
			'participation as a consolation.&rdquo;, or it can be used '
			'in a pejorative, non-relationally way, as in ' 
			'&ldquo;What a bunch of losers, I\'m embarrassed to be seen '
			'with them.&rdquo; '
			'We think the relational usage is more common, so judge '
			'&ldquo;loser&rdquo; to be '
			'relational'
			'.'
		)
	},
	'pair': {
		'query': 'pair',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Pair&rdquo; denotes a quantity or grouping, and '
			"doesn't express a relationship, so it is "
			'non-relational'
			'.'
		)
	},
	'supplier': {
		'query': 'supplier', 
		'correct': ['relational', 'partly-relational'],
		'response': (
			'&ldquo;Supplier&rdquo; is used to establish a relationship '
			'between companies, where the supplier supplies parts or '
			'materials to the other company (the relatum).  Therefore it '
			'is relational.  It can also be used in a generic non-relational '
			'sense, so we would also accept occasionally relational.'
		)
	},
	'energy': {
		'query': 'energy',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Energy&rdquo; does not indicate a relationship, so it '
			'is '
			'non-relational'
			'.'
		)
	},
	'rapport': {
		'query': 'rapport',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Rapport&rdquo; is a bridge-noun: '
			'&ldquo;the rapport between the students and teacher was '
			'breaking down.&rdquo;  Therefore it is '
			'non-relational'
			'.'
		)
	},
	'captain': {
		'query': 'captain', 
		'correct': 'partly-relational',
		'response': (
			'&ldquo;Captain&rdquo; can either refer to the rank, or to '
			'the leadership position in command of a ship. The former '
			'usage is non-relational, while the latter is relational. '
			"The relational usage doesn't seem distinctly more common, so "
			'we label &ldquo;captain&rdquo; '
			'occasionally relational'
			'.'
		)
	},
	'dispute': {
		'query': 'dispute', 
		'correct': 'non-relational',
		'response': (
			'&ldquo;Dispute&rdquo; is a bridge-noun: '
			'&ldquo;the dispute between business partners would not be '
			'resolved easily.&rdquo; Therefore it is '
			'non-relational'
			'.'
		)
	},
	'contract': {
		'query': 'contract',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Contract&rdquo; is a bridge-noun: '
			'&ldquo;the contract between the parties is binding&rdquo;. '
			'Therefore it is '
			'non-relational'
			'.'
		),
	},
	'reporter': {
		'query': 'reporter', 
		'correct': ['partly-relational', 'relational'],
		'response': (
			'&ldquo;Reporter&rdquo; is a role, which can certainly be '
			'used non-relationally, as in : &ldquo;What do I do for a '
			"living? I'm a reporter.&rdquo; "
			'However, it is very often used to designate a person&rsquo;s '
			'affiliation to a particular news organization, as in: '
			'&ldquo;Neha Thirani Bagri is a reporter with The New York '
			"Times.&rdquo; It isn't clear which is more "
			'common, so we consider &ldquo;reporter&rdquo; to be '
			'occasionally relational.  However, we also accept relational.'
		)
	},
	'gender': {
		'query': 'gender', 
		'correct': 'non-relational',
		'response': (
			"Gender is a property, so it is "
			"non-relational"
			'.'
		)
	},
	'reconnaissance': {
		'query': 'reconnaissance',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Reconnaissance&rdquo; does not indicate a '
			'relationship.  Therefore it is '
			'non-relational'
			'.'
		)
	},
	'tip': {
		'query': 'tip',
		'correct': 'partly-relational',
		'response': (
			'Tip is occasionally relational.  It can describe the furthest '
			'point along some extended part of an object.  Or it can refer '
			"to pointers, as in &ldquo;tips and tricks&rdquo;.  We don't "
			'think the '
			'relational usage is the most common, so we '
			'judge tip to be '
			'occasionally relational'
			'.'
		)
	},
	'war': {
		'query': 'war', 
		'correct': 'non-relational',
		'response': (
			'&ldquo;War&rdquo; is a bridge-noun: '
			'&ldquo;The war between the '
			'countries raged on for decades&rdquo;. '
			'Therefore it is '
			'non-relational'
			'.'
		)
	},
	'operator': {
		'query': 'operator', 
		'correct': ['partly-relational', 'relational'],
		'response': (
			'&ldquo;Operator&rdquo; is a role, designating someone '
			'controlling machinery or a production process. As such '
			'it establishes a relationship between the person and '
			'the thing they are operating.  It can also be used '
			'in a generic non-relational sense, e.g. &ldquo;'
			'Operators will receive a raise&rdquo; and it also '
			'appears in mathematical jargon, but we consider these '
			'usages to be less common.  We accept either '
			'relational or occasionally relational as correct '
			'responses.'
		)
	},
	'copper': {
		'query': 'copper', 
		'correct': 'non-relational',
		'response': (
			'&ldquo;Copper&rdquo; does not indicate a relationship.  '
			'Therefore it is '
			'non-relational'
			'.'
		),
	},
	'nail': {
		'query': 'nail', 
		'correct': 'non-relational',
		'response': (
			'&ldquo;Nail&rdquo; does not indicate a relationship.  '
			'Therefore it is '
			'non-relational'
			'.'
		),
	},
	'realization': {
		'query': 'realization',
		'correct': 'non-relational',
		'response': (
			'&ldquo;Realization&rdquo; does not indicate a relationship.  '
			'Therefore it is '
			'non-relational'
			'.'
		)
	},
	'writer': {
		'query': 'writer', 
		'correct': 'partly-relational',
		'response': (
			'&ldquo;Writer&rdquo; has a few meanings: it can refer to the '
			'vocation, it can '
			'refer to the role at a newspaper company, or it can refer to '
			'the person who wrote the screenplay of a movie (although '
			'screenwriter isolates that meaning). The first non-relational '
			'usage is probably the most common. The latter two usages, as '
			'roles, can be used relationally, and it is not excessively '
			'rare to see them used that way. '
			'Therefore, we consider &ldquo;writer&rdquo; to be '
			'occasionally relational'
			'.'
		)
	},
	'noon': {
		'query': 'noon',
		'correct': 'non-relational',
		'response': (
			'Noon is '
			'non-relational'
			' because it does not denote a relationship.'
		)
	},
	'head': {
		'query': 'head', 
		'correct': 'partly-relational',
		'response': (
			'&ldquo;Head&rdquo; has a few prominent meanings: it can '
			'signify the body part, it can signify a distal and bulbous '
			'part of an object, or it can signify a position of '
			'leadership.  The latter two meanings are relational, but the '
			'first meaning is probably more common.  We therefore label '
			'&ldquo;head&rdquo; as '
			'occasionally relational'
			'.'
		)
	},
	'neck': {
		'query': 'neck', 
		'correct': 'partly-relational',
		'response': (
			'&ldquo;Neck&rdquo; can mean the body part of an animal '
			'joining the head to the body, or it can mean an elongate '
			'constricted part of an object, such as a bottle or vase. '
			'The former usage is '
			'non-relational'
			', while the latter is '
			'relational.  We think that the latter usage is less '
			'frequent, so we label &ldquo;neck&rdquo; as '
			'occasionally relational'
			'.'
		)
	},
	'delegate': {
		'query': 'delegate',
		'correct': 'relational',
		'response': (
			'&ldquo;Delegate&rdquo; is '
			'relational'
			', because a delegate is defined in '
			'terms of that which it is appointed for (e.g. a delegate of '
			'the internal committee).'
		)
	},
	'investor': {
		'query': 'investor', 
		'correct': 'partly-relational',
		'response': (
			'&ldquo;Investor&rdquo; is a role. It can be used to '
			'designate someone who frequently invests in companies and '
			'projects generally, or it can be used to signify the '
			'relationship between the person who invests and the '
			'company / project '
			'in which she invests&mdash;the latter being a relational '
			'usage.  We '
			'think that the relational usage is roughly on par with the '
			'generic usage, and since it is not clearly more common, we '
			'choose to label &ldquo;investor&rdquo; as '
			'occasionally relational'
			'.'
		)
	},
	'athlete': {
		'query': 'athlete',
		'correct': 'non-relational',
		'response': (
			'Athlete is non-relational.  It is not essentially defined as a '
			"relationship to something / someone else, and isn't usually used "
			'to establish a relationship.'
		)
	},
	'mathematician': {
		'query': 'mathematician',
		'correct': 'non-relational',
		'response': (
			'Mathematician is '
			'non-relational'
			', it is not essentially '
			'defined as a relationship to something / someone else.'
		)
	},
	'scientist': {
		'query': 'scientist',
		'correct': ['non-relational', 'partly-relational'],
		'response': (
			'Scientist is non-relational.  It generally is used to refer to '
			'a vocation, without establishing a relationship to a particular '
			'thing.  However, it is possible to use to establish a relationship '
			'to an organization, as in &ldquo;She is a scientist from Bell '
			'Labs&rdquo;, so we would also accept occasionally relational.'
		)
	},
	'predecessor': {
		'query': 'predecessor',
		'correct': 'relational',
		'response': (
			'Predecessor is inherently '
			'relational'
			', being defined as the '
			'thing which came before something else.'
		)
	},
	'mayor': {
		'query': 'mayor',
		'correct': 'relational',
		'response': (
			'Mayor is '
			'relational'
			', expressing the relationship between a city and its '
			'administrative leader.'
		)
	},
	'blacksmith': {
		'query': 'blacksmith', 
		'correct': 'non-relational',
		'response': (
			'Blacksmith is '
			'non-relational'
			'.  Although it is a vocation, '
			"it isn't essentially defined in terms of a relationship to "
			'something else.'
		)
	},
	'partner': {
		'query': 'partner',
		'correct': 'relational',
		'response': (
			'Partner is inherently '
			'relational'
			', describing a person who '
			"is cooperating with someone else."
		)
	},
	'assistant': {
		'query': 'assistant',
		'correct': 'relational', 
		'response': (
			'Assistant is inherently '
			'relational'
			', describing a person who '
			"is supporting someone else."
		)
	},

	'colleague': {
		'query': 'colleague',
		'correct': 'relational',
		'response': (
			'Colleague indicates a peer relationship between people usually '
			'in a workplace.  Since it also refers to the relata, it is a '
			'relational noun.'
		)
	},
	'client': {
		'query': 'client',
		'correct': 'relational',
		'response': (
			'Client refers to the person or organization who receives a '
			'service offered by another person or organization.  Similar '
			'to customer, this directly indicates a relationship betweeen the '
			'party offering a service and the one receiving it.  Since it '
			'refers to one of the relata (the party receiving the service), '
			'it is relational.'
		)
	},
	'coach': {
		'query': 'coach',
		'correct': 'relational',
		'response': (
			'Coach indicates a relationship of leadership of a sports team '
			'similar to the relationship of an executive to his or her '
			'company.  Since it also refers to one of the relata (the '
			'leader), it is a relational noun.'
		)
	},
	'stepmother': {
		'query': 'stepmother',
		'correct': 'relational',
		'response': (
			'Stepmother is a simple case of a kinship term, so it is '
			'definitely relational.  It establishes a relationship between '
			"A child and the child's father's current wife, when that person "
			"is not the child's mother.  Since it refers to one of the relata "
			"(the father's wife) it is a relational noun)."
		)
	},
	'nominee': {
		'query': 'nominee',
		'correct': 'relational',
		'response': (
			'Nominee establishes a relationship between an award or position '
			'and a person who has been recommended for receipt of that '
			'award or position.  It refers to one of the relata (the person) '
			'so it is relational.'
		)
	},
	'viewer': {
		'query': 'viewer',
		'correct': ['relational', 'partly-relational'],
		'response': (
			'viewer establishes the relationship between a show (or other '
			'visual presentation or performance), and a person who '
			'observes it. Although we think this usage is dominant, it is also '
			'possible to use viewer in a generic sense, but even in those '
			'there is likely to be an implicit object of observation to which '
			'the relationship is established.  Therefore we consider viewer '
			'to be usually relational, but also accept occasionally relational.'
		)
	},
	
	'parent': {
		'query': 'parent',
		'correct': 'relational',
		'response': (
			'Parent establishes the relationship between one being and their '
			'offspring.  As a kinship noun it is clearly relational.'
		)
	},
	'admiral': {
		'query': 'admiral',
		'correct': ['relational', 'partly-relational'],
		'response': (
			'Admiral can be used relationally, in the sense that it '
			"establishes a relationship to a military's navy.  However being "
			'a rank as such is questionably relational, since we might feel '
			'that it is the rank alone that matters, and not which particular '
			'navy that the rank is held with.  We consider admiral to be '
			'occasionally relational, but also accept usually relational.'
		)
	},
	'estate': {
		'query': 'estate',
		'correct': 'relational',
		'response': (
			'Estate establishes the relationship between a person or '
			'family, and the sum total of their worldly objects, including '
			'especially properties.  It refers to the set of assets, which '
			'is one of the relata, so it is relational.'
		)
	},
	'back': {
		'query': 'back',
		'correct': ['partly-relational', 'relational'],
		'response': (
			'Back can refer either to portion of an object opposite its '
			'front (which is relational by vitue of being a relative part) '
			'or a body part of a person or animal accounting for most of the '
			"animal's posterior.  The body part meaning is probably not "
			'relational, but is so intertwined with the notion of the '
			'relative part that an argument could even be made for that usage '
			'being relational.  Since there is both a relational and an '
			'arguable non-relational usage, we accept both the '
			'usually relational and occasionally relational labels in this '
			'case.'
		)
	},
	'ammendment': {
		'query': 'ammendment',
		'correct': ['partly-relational', 'relational'],
		'response': (
			'Amendment can be an additional part added to a document or an '
			'alteration of a document, which qualifies as a relative part. '
			'It can also mean the act of ammending, which like most actions '
			"isn't considered relational for the purposes of this task. "
			'Given it has both relational and non-relational usages, we '
			'accept "usually relational" and "occasionally relational" here.'
		)
	},
	'alias': {
		'query': 'alias',
		'correct': 'relational',
		'response': (
			'"Alias", like the word "name", establishes a relationship '
			'between a person or other named thing, and a label or referrent '
			'used to identify it.  The only difference from "name" is that '
			'"alias" is understood to be an alternative label to the usual '
			'"name". Like the word "name", "alias" is a relational noun, '
			'because it establishes the referrer-referrent relationship and '
			'refers to one of the relata, the referrer.'
		)
	},
	'teammate': {
		'query': 'teammate',
		'correct': 'relational',
		'response': (
			'Teammate establishes the relationship between peers on the same '
			'team, and since it refers to those peers (i.e. to the relata) '
			'it is a relational noun.'
		)
	},
	'playmate': {
		'query': 'playmate',
		'correct': 'relational',
		'response': (
			'Playmate establishes the relationship between peers who are '
			'playing together. '
			'Since it refers to those peers (i.e. to the relata) '
			'it is a relational noun.'
		)
	},
	'nemesis': {
		'query': 'nemesis',
		'correct': 'relational',
		'response': (
			'Nemisis establishes a relationshp of extreme rivalry, and '
			'refers to the relata in that relationship, therefore it is '
			'relational.'
		)
	},
	'namesake': {
		'query': 'namesake',
		'correct': 'relational',
		'response': (
			'Namesake establishes the relationship between one person and '
			'another who has the same name as the first.  It refers to '
			'one of the relata (the second person introduced having the same '
			'name), so it is relational.'
		)
	},
	'founder': {
		'query': 'founder',
		'correct': 'relational',
		'response': (
			'Founder establishes the relationship between a company or other '
			'organization and its initiator(s). It refers to the initiator(s), '
			'who is a relatum in that relationship, hence is a relational '
			'noun.'
		)
	},
	'manager': {
		'query': 'manager',
		'correct': 'relational',
		'response': (
			'Manager is used to establish a relationship both between the '
			'people that they manage, and the resource or function that they '
			'manage, and as such it is relational.  It can also be used in '
			'a generic non-relational sense, as in &ldquo;Middle managers '
			'will be hit hard by the cutbacks.&rdquo;, but we think that the '
			'relational uses are much more common, so accept only '
			'the label &ldquo;usually&rdquo; relational here.'
		)
	},
	'contributor': {
		'query': 'contributor',
		'correct': 'relational',
		'response': (
			'Contributor establishes the relationship between some cause, '
			'project, or initiative, and a person who invests effort, or '
			'resources into it.  It refers specifically to the person, who is '
			'one of the relata in that relationship, and so it is relational.'
		)
	},
	'kin': {
		'query': 'kin',
		'correct': 'relational',
		'response': (
			'Kin is the most generic of the kinship nouns, which refers to '
			'a person or people who are related through decendence or '
			'marriage.  Like all kinship words, it establishes a relationship '
			'while referring to the relata, so it is relational.'
			
		)
	},
	'adviser': {
		'query': 'adviser',
		'correct': 'relational',
		'response': (
			"Adviser (or advisor) establishes the relationship  between a "
			'person who guides another, by offering advice. Since it refers '
			'to one of the relata (the person giving the advice), it is a '
			'relational noun.'
		)
	},
	'minder': {
		'query': 'minder',
		'correct': 'relational',
		'response': (
			'Minder refers to a person in charge of taking care of someone '
			'else, usually responsible for ensuring their safety. '
			'Given that it establishes a relationship between the person '
			'being minded and the person doing the minding, while referring '
			'to one of them, it is a relational noun.'
		)
	},
	'consultant': {
		'query': 'consultant',
		'correct': 'relational',
		'response': (
			'Consultant establishes a relationship between a person '
			'who offers temporary specialized services, often of an advisory '
			'nature, and an entity (usually a company) that receives the '
			'sevices.  As such, it establishes a relationship, while '
			'to one of the relata---the person offerring the services, '
			'so it is a relational noun.'
		)
	},
	'primier': {
		'query': 'primier',
		'correct': 'relational',
		'response': (
			'Premier refers to the head of some level of government '
			'administration, such as a district, province, or perhaps the '
			'nation, depending on the specific system of government. '
			'As such, it establishes the relationship between that '
			'administrative region and the person in charge of it, while '
			'refering to the person (a relatum), so it is relational.'
		)
	},
	'seatmate': {
		'query': 'seatmate',
		'correct': 'relational',
		'response': (
			'Seatmate refers to a person who sits next to another person, '
			'such as on a plane.  As such, it establishes a relationship '
			'between these people, while refering to them (the relata), so '
			'it is a relational noun.'
		)
	},
	'painter': {
		'query': 'painter',
		'correct': ['relational', 'partly-relational'],
		'response': (
			'Painter can refer to the person who painted a specific work, '
			'and as such establishes a relationship between the painter '
			'and the work, similar to "author".  While this usage is '
			'relational, it is quite common, perhaps more common, to use '
			'painter in a non-relational way, indicating simply someone who '
			'paints.  For this reason, we accept both "usually relational" '
			'and "occasionally relational".'
		)
	},
	'contractor': {
		'query': 'contractor',
		'correct': ['relational', 'partly-relational'],
		'response': (
			'Contractor can refer to the person who provides services to '
			'another, or who organizes and solicits companies for building '
			'in a construction project or other project.  These uses are '
			'relational.  However, contractor can also be used generically '
			'simply to mean one who tends to work offering their sevices '
			'by contract, rather than specifically establishing that role '
			'relative to a particular project or partner.  Since both '
			'the relational and non-relational uses seem prevelant, we accept '
			'both the "usually relational" and "occasionally relational" '
			'labels here.'
		)
	},
	'associate': {
		'query': 'associate',
		'correct': 'relational',
		'response': (
			'Similar to colleague, "associate" establishes a relationship '
			'between people who work together. Since it establishes a '
			'relationship while referring to the relata involved in the '
			'relationship, it is a relational noun.'
		)
	},

	'telephone': {
		'query': 'telephone',
		'correct': 'non-relational',
		'response': (
			'Telephnone refers simply to an object, without primarily '
			'playing the role of establishing any relationships, therefore '
			'it is non-relational.'
		)
	},
	'program': {
		'query': 'program',
		'correct': 'non-relational',
		'response': (
			'Program refers to an code artifact or an an organizations '
			"goal-oriented initiative.  In both cases,  it doesn't primarily "
			'establish a relationship, therefore it is non-relational.'
		)
	},
	'flour': {
		'query': 'flour',
		'correct': 'non-relational',
		'response': (
			'Telephnone refers to an object or substance, without primarily '
			'playing the role of establishing any relationships, therefore '
			'it is non-relational.'
		)
	},
	'norm': {
		'query': 'norm',
		'correct': 'non-relational',
		'response': (
			'A norm refers to a protocol or expected behavior.  While it '
			'certainly has to do with <span class="italic">how</span> '
			'relationships are negotiated, it does not itself establish '
			'a relationship, and so it is non-relational.'
		)
	},
	'vehicle': {
		'query': 'vehicle',
		'correct': 'non-relational',
		'response': (
			'Vehicle simply refers to an object whose basic function is '
			'locomotion, and is not used primarily to establish a '
			'relationship. Therefore it is non-relational.'
		)
	},
	'oversight': {
		'query': 'oversight',
		'correct': 'non-relational',
		'response': (
			'Oversight refers to the act of watching over something. '
			'While it does in a sense establish a relationship, this is '
			'somewhat incidental to its central meaning which identifies an '
			'act, and it does not refer relata, so it is non-relational.'
		)
	},
	'kiss': {
		'query': 'kiss',
		'correct': 'non-relational',
		'response': (
			'Kiss refers to an action. '
			'While, in context, it may very well establish a relationship, '
			'it does so through a chain of inference rather than the '
			'relationship being a core part of its meaning.  Furthermore, '
			'it does not refer relata, so it is non-relational.'
		)
	},
	'signing': {
		'query': 'signing',
		'correct': 'non-relational',
		'response': (
			'Signing refers to an action. '
			'While, in context, it may very well establish a relationship, '
			'it does so through a chain of inference rather than the '
			'relationship being a core part of its meaning.  Furthermore, '
			'it does not refer relata, so it is non-relational.'
		)
	},
	'oven': {
		'query': 'oven',
		'correct': 'non-relational',
		'response': (
			'Oven simply refers to an object based on its function, '
			'and is not used primarily to establish a '
			'relationship. Therefore it is non-relational.'
		)
	},
	'poker': {
		'query': 'poker',
		'correct': 'non-relational',
		'response': (
			'Poker refers to a specific game, or to a stick-like object '
			'used to poke. Neither of these uses functions to establish '
			'a relationship, nor do they refer to relata, so "poker" is '
			'non-relational.'
		)
	},
	'rock': {
		'query': 'rock',
		'correct': 'non-relational',
		'response': (
			'Rock simply refers to an object based on its function, '
			'and is not used primarily to establish a '
			'relationship. Therefore it is non-relational.'
		)
	},
	'purse': {
		'query': 'purse',
		'correct': 'non-relational',
		'response': (
			'Purse simply refers to an object based on its function, '
			'and is not used primarily to establish a '
			'relationship. Therefore it is non-relational.'
		)
	},
	'reverence': {
		'query': 'reverence',
		'correct': 'non-relational',
		'response': (
			'Reverence refers to the internal state of revering someone or '
			'something, or to the act of expressing this state.  While in '
			'context it certainly can be used to establish a relationship, '
			'it does not do so by refering to the relata of such a '
			'relationship, and so is non-relational.'
		)
	},
	'protestation': {
		'query': 'protestation',
		'correct': 'non-relational',
		'response': (
			'Protestation refers to an action. '
			'While, in context, it may very well establish a relationship, '
			'it does so through a chain of inference rather than the '
			'relationship being a core part of its meaning.  Furthermore, '
			'it does not refer relata, so it is non-relational.'
		)
	},
	'inhibition': {
		'query': 'inhibition',
		'correct': 'non-relational',
		'response': (
			'Inhibition refers to the quality of being inhibited, or '
			'restrained, but does not primarily function to establish a '
			'relationship.  Therefore it is non-relational.'
		)
	},
	'betrayal': {
		'query': 'betrayal',
		'correct': 'non-relational',
		'response': (
			'Betrayal refers to an action. '
			'While in context it is generally consequential to ones '
			'understanding of relationships, '
			'it does not itself refer to the relata of a relationship, '
			'so it is non-relational.'
		)
	},
	'orchestration': {
		'query': 'orchestration',
		'correct': 'non-relational',
		'response': (
			'Orchestration refers to the act of organizing. '
			'It does not centrally establish a relationship nor does it '
			'refer to relata, so it is non-relational.'
		)
	},
	'turn': {
		'query': 'turn',
		'correct': 'non-relational',
		'response': (
			'Turn refers to a decision point in some procedure, given to '
			'a particular actor.  While it does establish that actors role '
			'in some process, this is more a result of inference and '
			'establishing such a relationship is not centrally part of the '
			'meaning of the word.  Most importantly, turn does not refer to '
			'relata, so it is non-relational.'
		)
	},
	'game': {
		'query': 'game',
		'correct': 'non-relational',
		'response': (
			'Game simply refers to an object based on its function, '
			'and is not used primarily to establish a '
			'relationship. Therefore it is non-relational.'
		)
	},
	'consideration': {
		'query': 'consideration',
		'correct': 'non-relational',
		'response': (
			'Consideration refers either to the act of considering, '
			'or to the thought dedicated during such an act.  It can also '
			'refer to the sympathetic concern given to some cause or person. '
			'While in context consideration can be used to establish a '
			'relationship, its meaning is not centrally about a relationship '
			'and it does not refer to relata.'
		)
	},
	'malpractice': {
		'query': 'malpractice',
		'correct': 'non-relational',
		'response': (
			'Malpractice refers to the irresponsible actions of a '
			"professional that are not inline with the professional body's "
			'standards.  It does not primarily function to establish a '
			'relationship, nor does it refer to relata.'
		)
	},
	'kick': {
		'query': 'kick',
		'correct': 'non-relational',
		'response': (
			'Kick refers to a specific bodily action or gesture, and does '
			'not directly function to establish a relationship, therefore it '
			'is non-relational.'
		)
	},
	'statement': {
		'query': 'statement',
		'correct': 'non-relational',
		'response': (
			'Statement refers to a written or spoken utterance, i.e. an act '
			'of expressing language.  As such its meaning is not primarily '
			'centered around establishing a relationship, nor does it refer '
			'to relata, so it is non-relational.'
		)
	},
	'furniture': {
		'query': 'furniture',
		'correct': 'non-relational',
		'response': (
			'Game simply refers to an object, or collection of objects, '
			'based on their function, '
			'and is not used primarily to establish a '
			'relationship. Therefore it is non-relational.'
		)
	},

}

TERNARY_OPTIONS = [
	{
		'text': 'almost never relational',
		'class': 'non-relational'
	}, {
		'text': 'occasionally relational',
		'class': 'partly-relational'
	}, 
	{
		'text': 'usually relational', 
		'class': 'relational'
	}
]
BINARY_OPTIONS = [
	{'text': 'non-relational', 'class': 'non-relational'}, 
	{'text': 'relational', 'class': 'relational'}
]

AVAILABLE_RESPONSES = set([
	'non-relational', 'relational', 'partly-relational'])

def make_crowdflower_test_questions(out_path, batch_num):

	# Validate batch num, and get the correct batch of questions.
	if batch_num != 1 and batch_num != 0:
		raise ValueError('batch_num must be 0 or 1.')
	questions = (
		cf_test_questions_0 if batch_num == 0 else cf_test_questions_1
	)

	# Reproducibly randomize the order of the questions
	random.seed(0)
	random.shuffle(questions)

	# Open a csv writer
	out_f = open(out_path, 'w')
	writer = csv.writer(out_f)

	# Write the headings
	headings = [
		'token', 'source', '_golden', 'response_gold', 'response_gold_reason'
	]
	writer.writerow(headings)

	# The response map converts the naming of labels in question_specs to the
	# ones that are used in the crowdflower task
	response_map = {
		'non-relational': 'almost never relational',
		'partly-relational': 'occasionally relational',
		'relational': 'usually relational'
	}

	for question in questions:

		# The correct answer can either be a single string or a list thereof.
		# Normalize to be a list.
		correct = question_specs[question]['correct']
		if isinstance(correct, basestring):
			correct = [correct]

		# Ensure that the accepted response(s) are among the available responses
		for c in correct:
			if c not in response_map:
				raise ValueError((
					'The correct answer %s given for query %s are not all '
					'part of the available responses') % (c, question))

		correct = [response_map[c] for c in correct]
		correct = '\r\n'.join(correct)

		writer.writerow([
			question, 'test', 'true', correct,
			question_specs[question]['response']
		])




def make_quiz_questions(grouping, arity, retain=10):
	"""
	Make the HTML for practice questions appearing in the Crowdflower task.
	``grouping`` should be one of the keys in the global dict 
	``question_groupings``, which is used to select a subset of questions
	to be created.
	"""
	random.seed(0)
	questions_container = div()
	retain = retain or len(question_groupings[grouping])
	randomized_questions = random.sample(question_groupings[grouping], retain)
	print randomized_questions
	return
	for i, query in enumerate(randomized_questions):
		spec = question_specs[query]
		questions_container.appendChild(make_quiz_question(
			i, spec, arity, grouping
		))

	return unescape(questions_container.toprettyxml())


def unescape(string):
	return string.replace('&amp;', '&')


def make_quiz_question(i, spec, arity, grouping):
	question_wrapper = span({'class':'quiz'})

	# Make query part
	query_line = question_wrapper.appendChild(span({'class':'queryline'}))
	query = query_line.appendChild(span({'class': 'query'}))
	query_word = query.appendChild(span({'class':'query-word'}))
	query_word.appendChild(text(spec['query']))

	# Answer
	correct = query_line.appendChild(span({'class':'correct-answer'}))
	correct.appendChild(text(spec['correct']))

	# Make and append options
	option_wrapper = query_line.appendChild(div({'class':'option-wrapper'}))
	for option in make_options(i, arity, grouping):
		option_wrapper.appendChild(option)

	# Make response portion
	response_line = question_wrapper.appendChild(span({'class':'responseline'}))
	response = response_line.appendChild(span({'class':'response'}))
	prefix = response.appendChild(span({'class':'prefix'}))
	prefix.appendChild(text('prefix'))
	use_response = spec['response']
	response.appendChild(text(use_response))

	return question_wrapper


def make_options(i, arity, grouping):
	options = []

	option_types = BINARY_OPTIONS if arity=='binary' else TERNARY_OPTIONS
	for j, option in enumerate(option_types):
		option_elm = span({'class':option['class']+'-option option'})
		option_elm.appendChild(element(
			'input', 
			{
				'type':'radio',
				'id':'%s.%s.%s'%(grouping, i,j),
				'name':'%s.%s'%(grouping, i)
			}
		))
		label = option_elm.appendChild(element(
			'label', {'for':'%s.%s.%s'%(grouping,i,j)}
		))
		label.appendChild(text(option['text']))
		options.append(option_elm)

	return options


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

