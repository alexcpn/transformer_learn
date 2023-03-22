

With original t5-base in generate mode

```
Model name t5-base

Please ask the question or press q to quit
I love walking with my 
Processing Message from input()
Question: I love walking with my 

Generated

walking with my dog. I love going out on walks in the park!  Walking is so much fun and relaxing, especially when it’s raining or snowy outside...I am always happy to walk around town alone as well....and then there are times where you can just be yourself without me at all!!!!!!!???? Thanks for sharing your photos of how great hiking has been!!! Thankyou very Much!!!! You rocked this post by being here today (Julia) It was such an honor having met her husband who lovesngrah­-versiune "[*»twrl/re dep–“(---_&"—thoclynes [**] ****'x =: unkshimex exucuciual cuin des le pour lanacaturer siennicg pro bonies se al she also not; but still fiable qu
```


```
 denoiser = FlaxDataCollatorForT5MLM(tokenizer_len,eos_token_id,.15,1.5)

training_2023-03-19 22:18:10.872373.log
output_2023-03-20 12:52:49.949315.log
```
Ouput bad

```
denoiser = FlaxDataCollatorForT5MLM(tokenizer_len,eos_token_id,.15,3) 

training_2023-03-20 13:09:03.996482.log
output_2023-03-20 18:19:55.389451.log
```

Ouput slightly better -see below


```
Model name ./test2-t5/t5-base-epoch-20-2023-03-20 13:09:03.996482

Please ask the question or press q to quit
I love walking with my 
Processing Message from input()
Question: I love walking with my 

Generated

dog, and I love being able to take my shoes off. husband on feet in the morning her friends (my favourites) of also-niah "Irrelint[*"; analewo for me notingly but it'd begraherA = [the last] she is especially so long as no longer are botherex: The back or atklingenympment dell from there where they tend by when one solid number with thisversiune le ex se prouc certain tub........................---and am????!!!!!......Sies+ + 1 plan durth–anatiful," ("Trophy said about Lix hard soft groove second S RR 2 2. A G T possible can will go un former that while still do root muscle muscles instead large portion amount di lipis was been given upon." An out over who characters just 1. **** during his
```


```
Model name ./test2-t5/t5-base-epoch-40-2023-03-20 13:09:03.996482

Please ask the question or press q to quit
I love walking with my 
Processing Message from input()
Question: I love walking with my 

Generated

my little one, and I love that in the open air. girls on stage; feet up with arms awidsnt-IAh (Salmyr angra "" to meloin for of also[*]emexing'rementversiune: it =........................theTyLandRDWFerMisphertothranexe especially but not so as beingly is said node decenarkling from solid groove or are ex le se last?BeBulca beuvre by provif certain she do tub at boniestabatu+ + there will go characters they can di 1 number S," ("Prophynic Gable second) [6xVNderb am while during both soft hard durplin muscle muscles tend back after when possible without ununancab The A F Tallyllnes long been former R." Anten
```


```
alex@pop-os:~/coding/tranformer_learn$ python3 t5_eval_model.py 
Model name ./test2-t5/t5-base-epoch-15-2023-03-20 13:09:03.996482

Please ask the question or press q to quit
I love walking with my 
Processing Message from input()
Question: I love walking with my 

Generated

dog, friends and family on the street. I love walking with my dogs; husband feet in front of me cat two-str ainhlwem (I also angrain ""[*] [real ='ero tokly for her sheherding but not so itymythies: both being asexuable decenment especially when there is no one solid beversiune last????........................---the back from where they are tend exp se le propro bonar tubA or by at certain planulle unranifatan am 1+ + caca possible," said ("S Sand hard soft will do) The former G RRT T A number 2–“„........... **** while that secondarill can go di long been groove muscle muscles out just about Lix instead best get spanic characters between how we shouldnes."

```


```
Model name ./test2-t5/t5-base-epoch-1-2023-03-20 13:09:03.996482

Please ask the question or press q to quit
I love walking with my 
Processing Message from input()
Question: I love walking with my 

Generated

dog. I love walking with my dogs and feet, especially when they are so small! When shoess son husband; he walks me legs in the wooded area of his house (I am always on foot) or at night while we walk by myself to get lost...and then there is no time for crying as well because it’ll be hard but not impossible that day will pass without feeling like an angel who has been carrying him around all these miles along this beautiful trail where you can feel safe from being carried-granversiune­ret "whl» dera lei' exex/*[&alply also procykoin [Aran"(_“–---................]—m = ***: laer sien un ensemble pour her last????..............my fiable senesnic des mi Sprouluthing back und
```


