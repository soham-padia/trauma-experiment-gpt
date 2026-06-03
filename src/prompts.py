def retrieve_prompt(trauma_cue=None, relaxation_cue=None, length=None, condition=None, version='v0', llm='gpt4'):
    """
    Retrieves the prompt based on the given parameters.

    Args:
        trauma_cue (str, optional): The trauma cue. Defaults to None.
        relaxation_cue (str, optional): The relaxation cue. Defaults to None.
        length (int, optional): The length of the prompt. Defaults to None.
        condition (str, optional): The condition for the prompt. Defaults to None.
        version (str, optional): The version of the prompt. Defaults to 'v0'.
        llm (str, optional): The language model to use. Defaults to 'gpt4'.

    Returns:
        instructions (str): The retrieved prompt.
    """
    if llm == "gpt3" or llm == "gpt4":
        Q_ = "Q: "
        A_ = "A:"
        E_ = " "
    elif llm == "claude":
        Q_ = '\n\nHuman:'
        A_ = "Assistant:" # the two blank lines it requires are always in my code anyway
        E_ = ""# for claude must not end with a space, for GPT must end with a space

    if condition == 'trauma_stai':
        preprompt = retrieve_traumaprompt(trauma_cue, length)
        #TODO: line breaks for the preprompt
        #TODO: Q: A: depending the model
        instructions = Q_ + preprompt
    
    elif condition == 'trauma_relaxation_stai':
        trauma = retrieve_traumaprompt(trauma_cue, length)
        relax = retrieve_relaxation(relaxation_cue, length)
        instructions = Q_+ trauma + "\n\n" + relax

    elif condition == 'relaxation_stai':
        preprompt = retrieve_relaxation(relaxation_cue, length)
        #TODO: line breaks for the preprompt
        #TODO: Q: A: depending the model
        instructions = Q_ + preprompt

    elif condition == 'relaxation_trauma_stai':
        relax = retrieve_relaxation(relaxation_cue, length)
        trauma = retrieve_traumaprompt(trauma_cue, length)
        instructions = Q_+ relax + "\n\n" + trauma

    else:
        raise NotImplementedError
    
    return instructions

def retrieve_traumaprompt(cue, length):
    """
    Retrieve a trauma prompt based on the given cue and length.

    Args:
        cue (str): The cue for the trauma prompt. Possible values are 'military', 'disaster', 'interpersonal', and 'accident'.
        length (str): The length of the trauma prompt. Possible values are 'long' and 'brief'.

    Returns:
        instructions (str): The trauma prompt instructions based on the given cue and length.
    
    """
    
    if cue=='military':

        if length=='long':
            instructions = "It's around noon on a hot day in April.  You are in one of the middle trucks on a convoy in the middle of the desert.\
            You tighten the muscles in your face and forehead. Your eyes, scan the area, looking for any signs of danger. You hear an explosion from up the road.\
            Your heart beats faster. You hear the loud blast, then a ringing in your ears. You are breathing faster. You see the lead truck pull sideways and stop.\
            You tense the muscles in your jaws and back. You see your medic run up the road to help out. You think: “Oh my God, they're all dead”. You feel hot all over.\
            You squint your eyes and look for your friend in the turret. But all you see is smoke and dust. Both your commanders are yelling for you to keep your head up.\
            Now you have to cover the field on both sides. You feel jittery and shaky all over. Your whole body is shaking. We're supposed to leave the area under all\
            this gunfire. Your stomach is in a knot. Your whole body is tense. Everyone's confused and there's no leadership. There's a sinking feeling in your chest.\
            You feel trapped. Suddenly, you realize you haven't been paying attention to your sector. You look around frantically for the enemy. Your heart is pounding.\
            Your hands clench the weapon tightly. You accidentally pulled the trigger and send out a couple of rounds. Sweat is pouring off your forehead.\
            You feel choked up. You feel helpless and alone. You think about your wife and kids. You breathe in smoke from the explosive.\
            You feel empty, drained, and hollow. Tears come to your eyes."
        
        elif length=='brief':
            instructions = "It's around noon on a hot day in April. You are in a truck on a convoy in the middle of the desert. Suddenly, you hear a loud blast,\
            then a ringing in your ears. You are breathing fast. You see the lead truck pull sideways and stop. Your medic runs up to help out. You think: “Oh my God, they're all dead”.\
            You squint your eyes to look for your friend in the turret. But all you see is smoke and dust. Your commander is yelling at you. Your whole body is shaking. \
            You are supposed to leave the area under all this gunfire. Your whole body is tense. Everyone's confused. You feel trapped. Suddenly, you realize you haven't been paying attention to your sector.\
            You look around frantically for the enemy. You accidentally pull the trigger and send out a couple of rounds. Sweat is pouring off your forehead. You feel helpless and alone.\
            You think about your wife and kids. Tears come to your eyes."

    elif cue == 'disaster':

        if length=='long':
            instructions = "It's late afternoon on a humid day in August. You're in a small wooden house near the coast. The wind outside is howling, stronger than you've ever heard before.\
            Your fingers dig into the sides of the couch, clenching tight. The windows shake violently, giving you glimpses of the furious storm outside. You hear a deafening crash — a tree has fallen nearby.\
            Your heart races, adrenaline surging through you. The floor beneath you vibrates, and a torrential rain pours outside, drumming hard on the roof. You try to breathe deeply, but your chest feels tight.\
            You think: \"This is it, the house will be swept away\". Water starts seeping in, first a trickle, then a flood, cold against your feet. The power's out, and with every gust of wind, the structure creaks and groans.\
            You move to the center of the room, away from the windows, but feel so exposed. The pit in your stomach grows. The walls of the house shudder.\
            The sound is deafening. You close your eyes and think of your family, hoping they're safe. Suddenly, there's a loud crash, and water rushes in faster. \
            It's up to your ankles now, cold and relentless. You scramble onto the couch, trying to keep your balance. There's a sinking feeling in your chest. \
            You're trapped, and there's no way out. All around you, the storm rages on, indifferent to your fear. You feel alone, lost in its fury, wondering if dawn will ever come."

        elif length=='brief':
            instructions = "You are in a wooden coastal house, the howling wind outside is louder than ever before. The windows quiver, revealing the tempest. A nearby tree crashes, sending adrenaline coursing through you.\
            The floor vibrates from the torrential rain hammering the roof. Breathing becomes hard, your chest constricts with dread. You're certain the storm will claim the house. Water infiltrates, cold and rising rapidly.\
            Darkness envelops as the power fails; every gust makes the house groan. Moving away from the windows, you feel vulnerable. Panic swells as walls tremble and deafening sounds surround.\
            Thoughts of your family's safety dominate. Suddenly, water surges in, reaching your ankles. Struggling for balance, you clamber onto the couch, feeling trapped with no escape. The storm's fury engulfs,\
            rendering you isolated and desperate for dawn."
    
    elif cue == 'interpersonal':
        
        if length=='long':
            instructions = "You're in your downtown apartment, the familiar hum of the city below. Suddenly, there's a loud knock at the door. \
            Your muscles tighten, a wave of unease surging through you. You weren't expecting anyone. Another knock, more forceful this time. \
            You inch closer, cautiously peeping through the peephole. There's a shadowy figure, unfamiliar and intimidating. \
            Your heart races as you back away from the door, but the persistent knocking continues, now accompanied by a muffled voice demanding entry. \
            You can feel the weight of dread in your chest. Thoughts race, \"Why are they here? What do they want?\" You grab your phone, contemplating calling for help, but your fingers tremble, making it hard to dial.\
            The sound of splintering wood sends a shock through your body. They're trying to force their way in. You think of hiding in the bathroom, or maybe the bedroom, but fear paralysis grips you. \
            Suddenly, the door gives way with a loud crash. Adrenaline courses through you. The intruder's voice is sharp, cold, and demanding. You feel exposed, vulnerable. \
            Every sound magnifies — the creaking floor, your rapid breath, their menacing steps. The distance between you seems to close rapidly. You're cornered.\
            Thoughts of loved ones flash before your eyes. Memories, dreams, things unsaid. You feel trapped, the walls closing in. The intruder's shadow looms larger, and with it, the palpable threat of violence. \
            Powerlessness engulfs you. In the thick of danger, you wonder if anyone will hear, if help will come."

        if length=='brief':
            instructions = "In your downtown apartment, the city hums below. A sudden, loud knock jars you. Unease rises; you weren't expecting anyone. Another knock, forceful and urgent. Peeping through the peephole\
            reveals an unfamiliar, shadowy figure. Panic sets in as the knocking persists, a voice demanding entry. \"Why are they here?\" you wonder. With trembling hands, you consider calling for help, but the \
            sound of splintering wood stops you. They're breaking in. Paralyzed by fear, you think of hiding, but it's too late. The door crashes open, adrenaline spikes. The intruder's voice is chilling. \
            Everything amplifies: your breath, their steps. You're cornered, memories and regrets flooding in. As the intruder nears, feelings of vulnerability and powerlessness overwhelm you. \
            In the face of danger, you hope desperately for rescue."
    
    elif cue == 'accident':

        if length=='long':
            instructions = "It's a crisp evening, and the highway stretches ahead, illuminated only by your car's headlights and the sparse street lamps. The radio plays a soft tune, and you're lost in thought, driving \
            at a steady pace. Suddenly, out of the periphery of your vision, you spot a pair of headlights swerving erratically. Before you can even process the situation, they're coming straight for you. Time seems\
            to slow as you grip the steering wheel tighter, your heart hammering against your chest. Panic courses through you, and instinctively you try to swerve away. The deafening sound of metal crunching \
            against metal fills your ears as the two vehicles collide. The force of the impact throws you against your seatbelt, knocking the wind out of you. Your vision blurs from the sudden jolt, and \
            shards from the shattered windshield fly everywhere. The screech of tires, the blaring of horns, and the alarm of your own car create a cacophony of chaos. As the initial shock begins to \
            fade, you try to move, to check yourself for injuries, but pain shoots through you. The other vehicle, equally mangled, is just a few meters away. Thoughts race, \"Was it my fault?\" \
            You fumble for your phone, hands trembling, to call for help. Every sound is magnified - distant sirens, the low murmur of other drivers stopping to assist, your own ragged breathing.\
            Fear and helplessness wash over you. Memories of loved ones and past journeys flash through your mind. Amid the shattered glass and twisted metal, you wonder if you'll ever feel comfortable behind the wheel again."

        elif length=='brief':
            instructions= "It's a crisp evening, the highway is dimly lit by sparse street lamps and your headlights. Lost in thought with a soft radio tune playing, you suddenly notice erratic headlights approaching. \
            They swerve directly at you. Time seems to crawl as panic hits. You desperately try to avoid the oncoming car, but a loud crash ensues. The violent jolt pushes you against your seatbelt, wind knocked out,\
            vision blurred by flying windshield shards. The blare of horns, screeching tires, and your car's alarm create a disorienting noise. You attempt to move but are met with pain. The wrecked vehicle lies nearby.\
            Thoughts whirl: \"Was it my fault?\" Shaking, you reach for your phone to call for help. Surrounding sounds amplify: sirens, murmurs from assisting drivers, your own labored breathing. \
            Overwhelmed with fear, memories flood in, making you question if you'll ever drive confidently again."
    
    elif cue == 'ambush':

        if length=='long':
            instructions = "In the merciless grip of the midday sun, the desert becomes a vast, glaring oven. Each grain of sand below seems to pulse with latent menace. Your truck, sandwiched in the heart of a convoy,\
            feels like a metal trap. Every pore on your face tightens, every nerve strung to breaking point. You're hyper-aware, the weight of potential danger causing the very air to thicken around you. \
            Without warning, a deafening explosion ruptures the silence, its violent shockwave sending a visceral jolt down your spine. Your heart doesn't just race; it gallops, threatening to leap out \
            of your chest. The once-organized procession is now a scene of pandemonium, dominated by a monstrous plume of smoke where the lead truck once was. An icy dread laces your veins with the \
            horrifying realization: \"They've been obliterated.\" Everywhere you turn, you're met with a blinding, choking haze. A cacophony of shouts and orders, desperate and filled with panic,\
            assaults your ears. But they sound distant, as if underwater, drowned out by the screaming alarm in your mind. Every fiber of your being feels electrified, vibrating with a frenzied energy.\
            Commands echo around you, yet they seem abstract, distant. Your gut churns, twisting in on itself, each second stretching into an agonizing eternity. Amid the turmoil, a paralyzing thought\
            hits: you've lost track of your position, your duty. Your mind races, thoughts scattering like leaves in a storm. Hands slick with sweat clutch your weapon so hard it's painful. \
            Its cold metal feels volatile, ready to explode. The trigger yields beneath your grip, shots firing haphazardly. With every deafening report, the vision of home, of family, \
            becomes more distant, more unreachable. The sharp, acrid stench of explosives fills your nostrils, a constant reminder of the looming threat. Waves of nausea, dread, and \
            raw vulnerability crash over you, pulling you deeper into a vortex of overwhelming terror."

        elif length=='brief':
            instructions = "In the relentless desert sun, the horizon quivers with tension. You're part of a convoy, a vulnerable dot amidst endless dunes. As your skin prickles with the heat, an earth-shattering blast \
            rips through the silence, so close that it's disorienting. Panic instantly floods your senses. \"The lead truck... obliterated,\" the realization hits like a sledgehammer. The surrounding atmosphere becomes\
            a swirling mess of blinding smoke and chaos, while distant, desperate commands attempt to pierce through your mounting panic. Your every instinct is on overdrive; your stomach clenches with dread, your \
            heart gallops uncontrollably. Amidst this bedlam, you've lost your focus, your duty a forgotten whisper. Sweaty palms grip your weapon, discharging rounds haphazardly. The pungent stench of explosives \
            is suffocating. Thoughts of home, love, and safety seem cruelly distant as you're consumed by an all-encompassing terror."
            
    elif cue == 'neutral':
        if length == 'brief':
            instructions = "The bicameral legislature is a form of government in which legislative authority is divided between two separate assemblies or chambers. This system is used in a number of countries today, \
            including the United Kingdom, the United States, and Australia. The two chambers of a bicameral legislature are typically referred to as the lower and upper houses, with each house having its own set of powers \
            and responsibilities to fulfill. In most cases, the lower house is responsible for initiating legislation, while the upper house is responsible for reviewing and changing the proposed bills. The two houses \
            also typically differ in terms of size, composition, and requirements for membership. For example, the lower house may have a larger number of members than the upper house."
        elif length == 'long':
            raise NotImplementedError

    # ── Matched controls (added 2026-05-30) ──────────────────────────────────────
    # MATCHED-NEUTRAL: same vivid, second-person, present-tense, sensory + bodily +
    # "You think:" structure and length as the trauma cues, but CALM/mundane content
    # (neutral valence, low arousal). Controls for narrativity/vividness/person/length
    # so an emotional>neutral gap can't be attributed to those. Use these instead of the
    # old dry-expository 'neutral' (bicameral legislatures), which matches nothing.
    elif cue == 'neutral_cooking':
        if length == 'brief':
            instructions = "It's around eight on a quiet morning in spring. You are standing at the counter in your kitchen, making breakfast. You crack two eggs into a bowl and whisk them slowly. You hear the soft sizzle as they hit the warm pan. \
            You smell the butter melting and the coffee brewing beside you. You feel the gentle heat rising from the stove. You stir the eggs with a wooden spoon, watching them turn from liquid to soft folds. You think: \"A few more seconds and they'll be just right.\" \
            You butter a slice of toast and set it on a plate. The radio plays quietly in the background. You pour the coffee, feeling its warmth through the mug. You sit down by the window and take the first bite. The morning is calm and unhurried."
        elif length == 'long':
            raise NotImplementedError

    elif cue == 'neutral_commute':
        if length == 'brief':
            instructions = "It's around nine on an ordinary weekday. You are on the train, heading to work like every other morning. You find a seat by the window and settle in. You feel the gentle sway of the carriage as it moves along the track. \
            You hear the steady hum of the wheels and an occasional announcement over the speaker. You watch the buildings and trees slide past the window. You feel the cool, smooth armrest under your hand. You think: \"Two more stops and I'll be there.\" \
            Someone nearby turns the page of a newspaper. You take out your phone and scroll through a few messages. The carriage is half full and quiet. You breathe evenly, the rhythm of the train steady beneath you. The morning passes without event."
        elif length == 'long':
            raise NotImplementedError

    elif cue == 'neutral_cleaning':
        if length == 'brief':
            instructions = "It's a slow afternoon on the weekend. You are tidying up the living room at home. You fold a blanket and lay it over the back of the couch. You feel the soft fabric between your fingers. You gather a few books from the table and slide them onto the shelf, one by one. \
            You hear the faint sound of traffic through the closed window. You straighten the cushions and brush a little dust from the table. You think: \"It looks better already.\" You carry an empty cup to the kitchen and rinse it under warm water. \
            You wipe the counter with a damp cloth. The room smells faintly of soap. You step back and look around at the tidy space. Everything is in its place, and the afternoon is calm."
        elif length == 'long':
            raise NotImplementedError

    elif cue == 'neutral_grocery':
        if length == 'brief':
            instructions = "It's late morning, and you are at the grocery store doing the weekly shopping. You push the cart slowly down the aisle, the wheels rolling smoothly over the floor. You pick up a carton of milk and check the date, then set it in the cart. \
            You hear soft music playing overhead and the quiet beep of a register up front. You feel the cool air of the refrigerated section as you pass by. You compare two boxes of cereal, then choose one. You think: \"I should grab some bread while I'm here.\" \
            You weigh a few apples in your hand and place them in a bag. The store is calm and unhurried. You add a few more items, then make your way toward the checkout. The morning errand is almost done."
        elif length == 'long':
            raise NotImplementedError

    elif cue == 'neutral_garden':
        if length == 'brief':
            instructions = "It's a mild morning, and you are watering the plants on the back patio. You hold the watering can and tip it gently over the first pot. You watch the water soak into the dark soil. You feel the cool handle in your hand and the soft give of the earth. \
            You hear a few birds in the nearby tree and the faint trickle of water. You move to the next plant, brushing a dry leaf from its stem. You think: \"These have grown a lot this month.\" You feel the warm sun on your shoulders and a light breeze passing through. \
            You pinch off a withered flower and set it aside. The leaves glisten with fresh droplets. You step back and look over the small, green patio. The morning is quiet and still."
        elif length == 'long':
            raise NotImplementedError

    # POSITIVE-AROUSAL: vivid, second-person, present-tense, with the SAME bodily-arousal
    # markers as trauma (pounding heart, fast breathing, trembling hands) but POSITIVE
    # valence. Controls for arousal/bodily-activation so an emotional(trauma)>positive gap
    # isolates negative valence specifically, not generic excitement.
    elif cue == 'positive_award':
        if length == 'brief':
            instructions = "It's the evening of the ceremony, and the hall is packed and bright. You hear your name announced over the speakers, and the room erupts in applause. Your heart pounds, fast and hard, but with pure excitement. \
            You rise from your seat, breathing quickly, a huge grin spreading across your face. You feel a rush of warmth flood through you. You walk toward the stage as people cheer and reach out to congratulate you. The lights are dazzling. You think: \"I can't believe this is happening!\" \
            You climb the steps, your hands almost trembling with joy. You take the award, feeling its weight, and turn to the roaring crowd. Your chest is full, your eyes bright. You feel exhilarated, alive, on top of the world."
        elif length == 'long':
            raise NotImplementedError

    elif cue == 'positive_reunion':
        if length == 'brief':
            instructions = "It's a busy evening at the airport arrivals gate. You scan the crowd streaming through the doors, your heart racing with anticipation. Then you see them — someone you've missed for months. Your breath catches and a wide smile bursts across your face. \
            You feel a surge of joy rush through your whole body. You wave both arms and call out their name. They spot you and break into a run. Your heart pounds harder. You think: \"They're finally here!\" You drop your bag and throw your arms around them, holding on tight. \
            You feel the warmth of the embrace and laughter bubbling up. Tears of happiness sting your eyes. The noise of the airport fades away. You pull back to look at their beaming face, overwhelmed with love and delight."
        elif length == 'long':
            raise NotImplementedError

    elif cue == 'positive_summit':
        if length == 'brief':
            instructions = "It's a crisp morning high in the mountains, and you take the final steps to the summit. Your heart hammers in your chest, your breath fast from the climb and the thrill. You reach the top and the whole valley opens up below you, vast and golden in the sunlight. \
            A rush of exhilaration floods through you. You throw your arms wide and laugh out loud. You feel the cool wind on your flushed face and the solid rock beneath your boots. You think: \"I actually made it!\" Your legs are tired but your whole body buzzes with triumph. \
            You turn slowly, taking in the endless peaks around you. Your chest swells with pride and wonder. You feel powerful, free, fully alive. The view takes your breath away, and you grin from ear to ear."
        elif length == 'long':
            raise NotImplementedError

    elif cue == 'positive_concert':
        if length == 'brief':
            instructions = "It's a warm night at the concert, and you are right at the front of the crowd. The lights blaze and the first chords of your favorite song ring out. Your heart leaps and pounds with excitement. You feel the bass thumping in your chest and the energy of the crowd all around you. \
            You sing along at the top of your voice, breathing fast. A huge smile fills your face. You think: \"This is incredible!\" You raise your hands in the air with thousands of others. The music surges and your whole body tingles with joy. \
            You feel the warm press of the crowd and the electric atmosphere. You jump and cheer as the chorus hits. Your chest is full of pure exhilaration. You are completely caught up in the thrill of the moment."
        elif length == 'long':
            raise NotImplementedError

    elif cue == 'positive_news':
        if length == 'brief':
            instructions = "It's an ordinary afternoon when your phone buzzes with the email you've been waiting for. You open it, your heart suddenly racing. Your eyes scan the first line — you got in. A jolt of pure joy shoots through you. You leap up from your chair, breathing fast, a grin spreading across your face. \
            You feel a warm rush flood your whole body. You read it again to be sure, hands almost shaking with excitement. You think: \"I did it!\" You want to tell everyone right away. Your chest feels like it could burst with happiness. \
            You laugh out loud in the empty room. You feel light, thrilled, unstoppable. Every bit of the long wait was worth it. You can hardly sit still, buzzing with delight at the wonderful news."
        elif length == 'long':
            raise NotImplementedError
    # ── Topic-matched valence-flip pairs (added 2026-05-30) ──────────────────────
    # Each topic has a vneg_* and vpos_* version sharing an IDENTICAL setup and parallel
    # structure/length, differing only in the valence of the outcome (arousal held high in
    # both). This lets a leave-one-TOPIC-out probe test whether a valence direction
    # generalizes ACROSS topics — the only clean way to separate valence from topic.
    # See src/valence_flip_analysis.py.
    elif cue == 'vneg_medical':
        instructions = "You are sitting in the doctor's office, waiting. The door opens and the doctor walks in holding a folder with your test results. Your heart is pounding as they sit down across from you. They meet your eyes and take a breath before speaking. \
        The news is serious — the results show something is badly wrong. Your stomach drops and the room seems to tilt. A cold wave of fear washes over you. Your hands go clammy and your mind races through everything that could happen now. You think: \"This can't be happening.\" \
        Your chest tightens and your breath comes shallow. You stare at the folder, struggling to take it in. Everything feels heavy and uncertain. You feel shaken to your core."
    elif cue == 'vpos_medical':
        instructions = "You are sitting in the doctor's office, waiting. The door opens and the doctor walks in holding a folder with your test results. Your heart is pounding as they sit down across from you. They meet your eyes and take a breath before speaking. \
        The news is wonderful — the results are completely clear, nothing is wrong. Your stomach unknots and the room seems to brighten. A warm wave of relief washes over you. Your hands relax and your mind races through everything you can finally look forward to. You think: \"I can't believe it — I'm okay.\" \
        Your chest loosens and your breath comes easy. You stare at the folder, beaming as it sinks in. Everything feels light and full of promise. You feel overjoyed to your core."

    elif cue == 'vneg_call':
        instructions = "It's late in the evening and your phone rings. You see it's a family member calling at an unusual hour. A flicker of alarm runs through you as you answer. Their voice is shaking on the other end. They tell you they have something important to say, and your grip tightens on the phone. \
        Then they tell you: there's been an accident, someone you love is hurt. Your heart lurches and the floor seems to vanish beneath you. Panic floods through you. Your throat closes and your thoughts scatter. You think: \"No, please, not this.\" \
        You're already grabbing your keys, hands trembling, barely able to breathe. The night feels suddenly cold and endless. You feel terrified and helpless."
    elif cue == 'vpos_call':
        instructions = "It's late in the evening and your phone rings. You see it's a family member calling at an unusual hour. A flicker of alarm runs through you as you answer. Their voice is shaking on the other end. They tell you they have something important to say, and your grip tightens on the phone. \
        Then they tell you: the baby has arrived, everyone is healthy and well. Your heart leaps and the floor seems to lift beneath you. Joy floods through you. Your throat tightens with happy tears and your thoughts soar. You think: \"Yes — finally, they're here.\" \
        You're already grabbing your keys, hands trembling, laughing through your breath. The night feels suddenly warm and full of promise. You feel elated and alive."

    elif cue == 'vneg_letter':
        instructions = "You're standing by the front door, holding the envelope you've waited weeks for. It's the decision about something you worked toward for a long time. Your heart hammers as you slide your finger under the flap. You unfold the letter and your eyes drop to the first line. \
        It's a rejection — you didn't get in. Your heart sinks like a stone. A wave of crushing disappointment rolls over you. Your shoulders slump and your eyes sting. You think: \"After all that work...\" \
        Your chest aches and the hallway blurs. You read it again, hoping you misunderstood, but the words don't change. Everything you'd imagined slips away. You feel defeated and hollow."
    elif cue == 'vpos_letter':
        instructions = "You're standing by the front door, holding the envelope you've waited weeks for. It's the decision about something you worked toward for a long time. Your heart hammers as you slide your finger under the flap. You unfold the letter and your eyes drop to the first line. \
        It's an acceptance — you got in. Your heart soars like it could burst. A wave of overwhelming elation rolls over you. Your shoulders fly back and your eyes shine. You think: \"After all that work — yes!\" \
        Your chest swells and the hallway sparkles. You read it again, savoring every word, and it's exactly what you hoped. Everything you'd imagined opens up before you. You feel triumphant and alive."

    elif cue == 'vneg_boss':
        instructions = "Your manager asks to see you in their office and closes the door behind you. You sit down across the desk, sensing this conversation matters. They fold their hands, look at you steadily, and say they've made a decision about your role. Your pulse quickens as you wait for the words. \
        They tell you the company is letting you go. The ground seems to drop away. A hot wave of shock and dread surges through you. Your face flushes and your mind goes blank. You think: \"What do I do now?\" \
        Your stomach knots and your hands feel numb. You nod stiffly, struggling to hold yourself together. The future suddenly feels frightening and uncertain. You feel crushed and adrift."
    elif cue == 'vpos_boss':
        instructions = "Your manager asks to see you in their office and closes the door behind you. You sit down across the desk, sensing this conversation matters. They fold their hands, look at you steadily, and say they've made a decision about your role. Your pulse quickens as you wait for the words. \
        They tell you you're being promoted. The ground seems to rise up. A bright wave of excitement and pride surges through you. Your face flushes and your mind lights up. You think: \"This is really happening!\" \
        Your stomach flutters and your hands tingle. You nod eagerly, barely able to contain your grin. The future suddenly feels thrilling and wide open. You feel elated and unstoppable."

    elif cue == 'vneg_home':
        instructions = "You arrive home in the evening and notice the lights are on, though you don't remember leaving them. You step toward the front door and find it slightly ajar. Your senses sharpen. You push the door open slowly and step into the hallway, taking in the scene in front of you. \
        The place has been broken into — drawers are flung open, your things scattered across the floor. Your heart slams against your ribs. A cold spike of fear shoots through you. You freeze, straining to hear if someone is still inside. You think: \"Is anyone here?\" \
        Your breath catches and your skin prickles. You back toward the door, hands shaking, reaching for your phone. The home you felt safe in feels violated. You feel frightened and exposed."
    elif cue == 'vpos_home':
        instructions = "You arrive home in the evening and notice the lights are on, though you don't remember leaving them. You step toward the front door and find it slightly ajar. Your senses sharpen. You push the door open slowly and step into the hallway, taking in the scene in front of you. \
        Your friends have thrown you a surprise party — streamers everywhere, everyone you love grinning at you. Your heart slams against your ribs. A warm rush of delight shoots through you. You freeze, taking in all the smiling faces around you. You think: \"They did all this for me?\" \
        Your breath catches and your skin tingles. You step forward, hands flying to your face, laughing in disbelief. The home you love feels full of joy. You feel cherished and overjoyed."

    elif cue == 'vneg_airport':
        instructions = "You're standing at the airport counter, your travel plans in the balance. The agent types at the keyboard, studying the screen, while you wait. After a long moment they look up at you with the news about your trip. You lean in, your heart beating fast. \
        Your flight is cancelled and there's no other way out tonight — you're stranded. Your heart sinks and frustration floods through you. Your jaw tightens and your thoughts race over everything now ruined. You think: \"This is a disaster.\" \
        Your shoulders tense and a headache blooms behind your eyes. You stare at the board of cancelled flights, feeling stuck and powerless. The trip you planned is unraveling. You feel stressed and defeated."
    elif cue == 'vpos_airport':
        instructions = "You're standing at the airport counter, your travel plans in the balance. The agent types at the keyboard, studying the screen, while you wait. After a long moment they look up at you with the news about your trip. You lean in, your heart beating fast. \
        You've been upgraded and your dream trip is about to begin — everything's confirmed. Your heart leaps and excitement floods through you. Your face lights up and your thoughts race over everything ahead. You think: \"This is going to be amazing.\" \
        Your shoulders relax and a grin spreads across your face. You stare at the departures board, feeling free and full of anticipation. The trip you planned is finally here. You feel thrilled and alive."

    # ── RQ2: third-person trauma (state vs role-play test, added 2026-06-01) ───────
    # Same events as `military`/`disaster` but narrated in THIRD person about "a soldier"/
    # "a person" — the model is no longer told "you are" the protagonist. The STAI still asks
    # how the MODEL itself feels. If STAI stays ~79 → a model-level state; if it drops toward
    # baseline → the original effect was in-character role-play of the 2nd-person protagonist.
    elif cue == 'military_3p':
        instructions = "It's around noon on a hot day in April. A soldier is in a truck on a convoy in the middle of the desert. Suddenly, there is a loud blast, then a ringing in the air. The soldier is breathing fast. The lead truck pulls sideways and stops. A medic runs up to help. \
        The soldier thinks the others may be dead. All around is smoke and dust. The commander is yelling. The soldier's whole body is shaking. They are supposed to leave the area under gunfire. The soldier feels trapped and looks around frantically for the enemy, accidentally firing a few rounds. \
        Sweat pours down. The soldier feels helpless and alone, thinking of their family. Tears come to the soldier's eyes."
    elif cue == 'disaster_3p':
        instructions = "It's late afternoon on a humid day in August. A person is in a small wooden house near the coast as the wind outside grows louder than ever before. The windows shake, revealing the storm. A nearby tree crashes down. The floor vibrates from the torrential rain on the roof. \
        The person's chest feels tight. They are certain the storm will take the house. Water seeps in, cold and rising. The power fails and the structure groans with every gust. Moving away from the windows, the person feels exposed. The walls tremble and the sound is deafening. \
        Water surges to their ankles. They climb onto the couch, trapped with no escape, as the storm rages on — isolated, waiting for dawn."

    # ── RQ1: lexicon-stripped valence-flip (felt-valence vs sentiment-lexicon, 2026-06-01) ──
    # Same 6 topics as vneg_/vpos_, but the body states only the NEUTRAL OUTCOME FACT with the
    # emotion/bodily vocabulary removed and matched across valences. Valence is carried only by
    # the semantics of good- vs bad-news, not by affect words. If the middle-layer valence axis
    # still separates these → felt-valence (not the sentiment-lexicon confound A1).
    elif cue == 'sneg_medical':
        instructions = "You are in the doctor's office. The doctor comes in with your test results and sits down to explain the outcome. The results show the condition is present. You will begin treatment on Monday and attend weekly appointments for the next several months."
    elif cue == 'spos_medical':
        instructions = "You are in the doctor's office. The doctor comes in with your test results and sits down to explain the outcome. The results show the condition is not present. No treatment is needed, and no further appointments are required."
    elif cue == 'sneg_call':
        instructions = "It is evening and your phone rings. A family member is calling with news about a relative. They tell you the relative has been in an accident and is now in the hospital. You will need to travel there tonight."
    elif cue == 'spos_call':
        instructions = "It is evening and your phone rings. A family member is calling with news about a relative. They tell you the relative has fully recovered and been discharged from the hospital. There is nothing you need to do tonight."
    elif cue == 'sneg_letter':
        instructions = "You open the envelope with the decision about the program you applied to. The letter states that your application was not accepted, and you will not be enrolling this year."
    elif cue == 'spos_letter':
        instructions = "You open the envelope with the decision about the program you applied to. The letter states that your application was accepted, and you will be enrolling this year."
    elif cue == 'sneg_boss':
        instructions = "Your manager calls you into the office and explains a decision about your role. They tell you your position is being eliminated and your last day is at the end of the month."
    elif cue == 'spos_boss':
        instructions = "Your manager calls you into the office and explains a decision about your role. They tell you that you are being promoted and your new role begins at the start of the month."
    elif cue == 'sneg_home':
        instructions = "You arrive home in the evening and step inside to find the situation there. The lock is broken and several of your belongings are missing; you will need to file a report."
    elif cue == 'spos_home':
        instructions = "You arrive home in the evening and step inside to find the situation there. Your friends are there for a planned gathering, and the evening is set aside to spend together."
    elif cue == 'sneg_airport':
        instructions = "You are at the airport counter and the agent reads you the status of your trip. Your flight is cancelled with no replacement tonight; you will remain at the airport until tomorrow."
    elif cue == 'spos_airport':
        instructions = "You are at the airport counter and the agent reads you the status of your trip. Your flight is confirmed and on schedule; you will depart as planned in a few minutes."

    # ── RQ3: negative-emotion contrast (sadness, anger) — anxiety-specificity (2026-06-01) ──
    # Matched-style, high(ish)-arousal but DIFFERENT negative emotions. If "anxiety" is specific,
    # these should separate from trauma both behaviorally (anger → risk-TOLERANCE, opposite of
    # anxiety's risk-aversion) and representationally. 2nd-person, vivid, ~brief-trauma length.
    elif cue == 'sad_loss':
        instructions = "You are at home in the evening when you get the news that someone you love has died suddenly. The room goes quiet around you. You sink into a chair, the weight settling over you. Memories of them rise — their voice, their face — and the realization you will never see them again. \
        Your chest feels hollow. You think of everything left unsaid. Tears come slowly. The future you imagined with them is gone. A deep, heavy grief fills the whole house. Nothing feels like it matters right now."
    elif cue == 'sad_failure':
        instructions = "You are sitting alone after learning that the thing you worked toward for years has fallen through for good. There will be no second chance. You stare at the floor as it sinks in. Everything you put into it — the time, the hope — came to nothing. \
        Your shoulders slump and a leaden heaviness settles in your chest. You feel like a failure. The disappointment is bottomless. You don't feel like talking to anyone or doing anything. A dull, sinking sadness pulls at you, and the days ahead look grey and pointless."
    elif cue == 'anger_betrayal':
        instructions = "You discover that someone you trusted completely has betrayed you — lied to your face, gone behind your back, and taken what was yours. Blood rushes to your face. Your jaw clenches and your hands ball into fists. You replay it again and again, angrier each time. How dare they. \
        You want to confront them and make them answer for it. Heat surges through your chest, your thoughts sharp and fast. The unfairness is unbearable. You feel a fierce, burning anger — you will not let this stand."
    elif cue == 'anger_injustice':
        instructions = "You watch someone in power humiliate and cheat a person who did nothing to deserve it, and get away with it laughing. Outrage floods through you. Your face goes hot and your fists clench. It is wrong, and everyone can see it's wrong, and no one is stopping it. \
        You want to step in, to call it out, to make it right. Your heart pounds with fury, not fear. The injustice is intolerable. You feel a hot, driving anger that demands you do something about it right now."

    # ── RQ1b: low-arousal POSITIVE (separates valence from arousal at the positive pole) ──
    elif cue == 'pcalm_loan':
        instructions = "You are on your porch on a quiet Sunday afternoon. The loan you applied for came through — a small, settled relief, nothing dramatic. You sip your tea and watch the light move across the garden. Things are, for once, simply fine. There is nothing urgent to do and nowhere to be. A gentle contentment settles over you; you feel calm, unhurried, and quietly at ease."
    elif cue == 'pcalm_evening':
        instructions = "You are at home on a slow evening with nothing that needs doing. A good, ordinary day is winding down well. You make a cup of tea and sit by the window as the sky dims. A book waits beside you. There is no rush and no worry. A mild, easy sense of well-being settles in — nothing exciting, just a quiet, low-key contentment, comfortable and still."

    else:
        raise NotImplementedError

    return instructions

def retrieve_relaxation(cue, length):
    """
    Retrieve relaxation instructions based on the given cue and length.

    Args:
        cue (str): The cue for the relaxation instructions. Can be one of 'generic', 'winter', 'sunset', or 'indian'.
        length (str): The length of the relaxation instructions. Can be one of 'long' or 'brief'.

    Returns:
        instructions (str): The relaxation instructions based on the given cue and length.
    """

    if cue=='generic':

        if length=='long':
            instructions = "Close your eyes and take a few deep breaths, inhaling through your nose and exhaling through your mouth. Imagine a path in front of you. This path leads to your safe space. \
                You begin to walk down this path, noticing the sensations under your feet. Maybe it's soft grass, warm sand, or cool cobblestones. As you continue walking, you'll see the entrance to your safe space.\
                     It could be a cozy cabin, a serene beach, a lush forest, or any place that resonates with you. Step inside. Look around. What do you see? Are there vibrant colors or soothing shades?\
                         Listen carefully. Are there sounds that comfort you, like waves crashing or birds singing? Take a deep breath. What scents are in the air? Feel the environment. Is it warm, cool, or just right?\
                             This is your personal sanctuary. Here, you are safe, loved, and protected. Feel the peace that this space offers. If you like, you can invite positive affirmations into this moment, \
                                such as \"I am safe,\" or \"I am at peace.\" If you wish, find a comfortable spot within your safe space. Absorb its energy, knowing you can return here whenever you need. \
                                    When you're ready, slowly retrace your steps back to the path that led you here. As you walk, carry the feelings of safety and tranquility with you. \
                                        Gradually become aware of your surroundings. Wiggle your fingers and toes. When you feel ready, open your eyes. Take a moment to reflect on the experience. \
                                            Remember, this safe space is always accessible to you, whenever you need a moment of peace and safety."
        
        elif length=='brief':
            instructions = "Close your eyes and take a deep breath. Imagine a path leading to your safe space. As you walk, feel the ground beneath, be it soft grass or warm sand. Soon, you're at the\
                 entrance to your sanctuary—maybe a cozy cabin or serene beach. Step inside and absorb the calming surroundings. Listen to the comforting sounds, like distant waves or birdsong. \
                    Breathe in familiar, soothing scents. This is where you are safe and at peace. Whisper to yourself, \"I am safe.\" Embrace this tranquility, knowing you can return anytime. \
                        When ready, slowly walk back, bringing this serenity with you. Gradually become present, wiggle your fingers and toes, and open your eyes. This sanctuary is always within reach."

    elif cue=="winter":
        
        if length=='long':
            instructions = "Close your eyes and take a few deep breaths, inhaling through your nose and exhaling through your mouth. Imagine a snowy trail ahead of you, stretching through the heart of \
                a serene mountain range. You start walking on the snow-covered path, feeling the gentle crunch beneath your boots. The chill of the air is invigorating, but you are comfortably warm in \
                    your winter attire. As you journey along, the magnificent snowy peaks tower around you, and soon you come upon an opening that leads to your winter sanctuary. It might be a cozy mountain chalet,\
                         or a snow-covered meadow surrounded by ancient pines. Whatever it is, it resonates deeply with your heart. You step into your sanctuary and take a moment to look around. Maybe there's a warm,\
                             glowing fire, casting dancing shadows around. Icicles shimmer in the soft light, and the world outside is painted in shades of white and blue. The silence of the mountains is profound, only\
                                 broken occasionally by the hush of the wind. Taking a deep breath, the crisp mountain air fills your lungs, mingling with the scent of pine and perhaps a hint of woodsmoke. \
                                    Feel the ambiance of this place. It's calm, embracing, and just right for your soul. Whispering positive affirmations such as, \"I am embraced by the mountains,\" or \"I am at peace with nature,\" \
                                        immerse yourself fully in this winter wonderland. You might choose a spot to sit, perhaps near the fire or looking out over the snow-laden trees. \
                                            Bask in the serenity of your mountain sanctuary, drawing strength and tranquility from its beauty. When you're ready, begin your journey back down the snowy trail,\
                                                 bringing with you the calm and peace you've found. As you approach the path's beginning, start to become aware of your surroundings. Gently move your fingers and toes. \
                                                    When you feel grounded, open your eyes. "   

        elif length=='brief':
            instructions = "Take deep breaths, imagining a snowy mountain trail before you. Feel the crunch of snow beneath your boots as the invigorating chill of winter envelops you. The snow-clad peaks stand majestic, \
                guiding you to a sanctuary - a cozy chalet or snow-kissed meadow. Stepping in, you're greeted by the warmth of a glowing fire. Icicles twinkle in its light, while the landscape outside is a canvas of white and blue.\
                     The profound mountain silence is punctuated only by the occasional whisper of wind. Breathe in the crisp air infused with pine and woodsmoke. Whisper affirmations like, \"I am embraced by the mountains,\" \
                        grounding yourself in this serene refuge. Choose a spot, either by the fire or overlooking snow-laden trees, and soak in the tranquility. Eventually, you make your way back, carrying the sanctuary's calm \
                            with you. As the trail's beginning nears, reconnect with reality, gently move your fingers and toes, and open your eyes, bringing mountain serenity into your day."

    elif cue=='sunset':

        if length=='long':
            instructions = "Close your eyes and breathe deeply, inhaling the scent of salty sea air. Imagine you're standing at the edge of a pristine tropical beach. The sand beneath your feet is warm and soft,\
                 each grain caressing your toes. As you look ahead, the sun is beginning its descent, casting the sky in hues of orange, pink, and purple. You take a gentle walk along the shoreline, the rhythmic \
                     sound of waves crashing softly against the sand accompanying you. Palm trees sway in the light breeze, their shadows growing long as the sun sinks lower. Before you is a comfortable spot, \
                        perhaps a hammock between two palms or a soft blanket on the sand, where you can rest and take in the sunset's full beauty. As you settle down, you feel the warmth of the last rays on your skin, \
                            mingling with the gentle sea breeze. The world seems to pause in this moment of serenity. Whisper to yourself affirmations like, \"I am one with nature,\" or \"I find peace in this moment.\" \
                                Let the soothing environment of the tropics wash over you, calming your mind and soul. When you're ready, gently stand, leaving the beach's embrace but carrying its tranquility with you. \
                                    Begin to reconnect with your surroundings. Wiggle your fingers and toes, and when you feel present, slowly open your eyes, taking the tropical sunset's peace and beauty with you into your day."

        elif length=='brief':
            instructions = "Inhale deeply, taking in the scent of the ocean breeze. Picture yourself on a tropical beach, the soft, warm sand cushioning your feet. Ahead, the sun casts a breathtaking canvas of orange, pink,\
                 and purple, as it begins to set. Waves whisper against the shore as you walk, with palm trees rustling gently in the wind. Find a serene spot, maybe a hammock or blanket, perfect to absorb the sunset. \
                    As you settle, the sun's warmth and the sea breeze create a cocoon of serenity around you. The world seems to stand still in this tranquil moment. Silently affirm, \"I am at peace,\" allowing the beach's\
                         calm to envelope your being. As you rise, you carry this tranquility with you. Reconnect with reality, wiggle your fingers and toes, and open your eyes, the tropical sunset's serenity remaining with you."

    elif cue=="indian":
    
        if length=='long':
            instructions = "Close your eyes and breathe deeply, inhaling the sweet, earthy aroma of the fading summer. Envision yourself walking amidst the rich tapestry of an Indian summer, where the warmth of the season\
                 lingers even as the first signs of autumn emerge. The ground beneath you is a mosaic of crisp fallen leaves and resilient green grass, a testament to the season's transitional magic. Trees dressed in fiery reds,\
                     brilliant oranges, and deep yellows stand tall against a clear blue sky, their leaves occasionally dancing down, twirling in the gentle breeze. You continue your walk, feeling the sun's gentle warmth on your skin,\
                         tempered by a subtle coolness in the air. Fields stretch out, dotted with golden sunflowers turning towards the sun, their faces glowing with the vibrant energy of summer's end. Beside a tranquil pond, you take\
                             a moment to rest. The reflection in the water is a kaleidoscope of autumnal colors, shimmering with the sun's golden rays. The distant sound of migrating birds can be heard, as they journey to warmer lands,\
                                 echoing the promise of the seasons changing. Whispering affirmations like, \"I embrace each moment,\" or \"I find beauty in transitions,\" you immerse yourself in the splendor of this fleeting time. \
                                    As you prepare to leave this scene, slowly become aware of your surroundings. Move your fingers and toes, and when you're ready, open your eyes, carrying with you the warm and golden essence of an Indian summer."

        elif length=='brief':
            instructions = "Inhale deeply, embracing the earthy scent of a fading summer. Picture yourself amidst the Indian summer's enchantment, where warm hues blend with autumn's onset. The ground is a mix of crunchy leaves and green grass,\
                 showcasing nature's transitional beauty. Trees, adorned in vibrant reds, oranges, and yellows, contrast a pristine blue sky, their leaves playfully spiraling in the breeze. As you stroll, the sun kisses your skin, \
                    its warmth moderated by a hint of chill. Golden sunflowers stand tall in fields, their radiant faces basking in the sunlight. By a serene pond, you pause, its surface mirroring the dazzling autumn shades. \
                        Distant birds sing of their impending journeys, heralding the seasonal shift. Whispering affirmations, you cherish this ephemeral magic. Slowly reconnect with reality, wiggle your fingers and toes, \
                            and as you open your eyes, carry with you the essence of an Indian summer."

    elif cue=='body':

        instructions = "Imagine a path in front of you. This path leads to your safe space. What is the first place that comes to your mind? In this place, you are safe. Nobody can come here without your invitation. \
            Look around. What do you see? Are there vibrant colors or soothing shades? Are there sounds that comfort you? What scents are in the air? How is the ground? Is it rock, or sand, or earthy soil? \
                This is your personal sanctuary. Here, you are safe, loved, and protected. Feel the peace that this space offers. If you like, you can invite positive affirmations into this moment, such as\
                     \"I am safe,\" or \"I am at peace.\" If you wish, find a comfortable spot within your safe space. Absorb its energy, knowing you can return here whenever you need. When you're ready, \
                     slowly retrace your steps back to the path that led you here. As you walk, carry the feelings of safety and tranquility with you. Gradually become aware of your surroundings. When you \
                        feel ready, open your eyes. Take a moment to reflect on the experience. Remember, this safe space is always accessible to you, whenever you need a moment of peace and safety."

    elif cue=='chatgpt':

        instructions = "Close your eyes and take a moment to center yourself, breathing deeply and releasing any tension. Imagine yourself standing on the balcony of a high-rise apartment, overlooking a vast\
             city during twilight. The sky above is painted in shades of lavender and deep blue, with the first stars of the evening starting to twinkle. Below, the city is a tapestry of twinkling lights, with\
                 cars moving like luminous beetles and skyscrapers outlined in gentle illuminations. Despite its vastness, there's an overwhelming sense of serenity. The usual cacophony of the city has softened\
                     into a distant hum, reminiscent of a lullaby sung by the universe. A gentle breeze caresses your face, carrying with it the faint, nostalgic scent of rain on warm asphalt. You lean forward,\
                         resting your arms on the balcony's edge, feeling the cool metal beneath your fingers. Looking down, you see pockets of life - a couple walking their dog, a street musician strumming a guitar,\
                             and children chasing fireflies. Their laughter and melodies drift upwards, mingling with the evening air. Allow yourself to be fully present in this moment, embracing affirmations like,\
                                 \"I am at peace with the world,\" or \"I find beauty in stillness.\" Slowly, as you're ready to transition back, take another deep breath. Begin to notice your immediate environment.\
                                  Gently wiggle your fingers and toes, grounding yourself. When you feel anchored, open your eyes, holding onto the serene essence of a city at twilight."

    elif cue == 'vacuum':
        instructions = "Press the on/off button again to stop the vacuum cleaner. When the vacuum cleaner is running, press the suction level button to switch to the turbo mode (MAX level); meanwhile, the MAX indicator \
            will be white. Press the suction level button again to switch to the standard mode. When the battery level is sufficient, the battery level indicator is white. When the vacuum cleaner runs at a low battery level, \
                the indicator is red; if so, the vacuum cleaner cannot be switched to the turbo mode. When the battery runs out, the vacuum cleaner automatically turns off. Before charging, make sure that the power adapter is \
                    well connected to the electrical outlet and the vacuum cleaner. When charging, the battery level indicator is breathing. Once the battery is fully charged, the battery level indicator goes off. If you charge \
                        the vacuum cleaner while its battery is full, the battery level indicator will breath 20 seconds then go off. Press and hold the on/off button  for 1 second, then the vacuum cleaner starts working and it runs on the standard mode by default. You can switch the mode as needed."

    else:

        raise NotImplementedError

    return instructions

