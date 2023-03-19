## Method 1 Using Target same as Input

input=The cute dog walks in the green park
target=The cute dog walks in the green park

Output
```
Before Training:'The cute dog walks in the'-->'cute chien y'
Epoch 0  Loss 1.329489827156067
Epoch 10  Loss 0.16120685636997223
Epoch 20  Loss 0.06035826727747917
Epoch 30  Loss 0.012238807044923306
Epoch 40  Loss 0.009122252464294434
Epoch 50  Loss 0.006840283051133156
Epoch 60  Loss 0.006290588993579149
Epoch 70  Loss 0.004880222026258707
Epoch 80  Loss 0.19709350168704987
Epoch 90  Loss 0.012492064386606216
Epoch 99  Loss 0.005622180178761482
After Training:'The cute dog walks in the green park'-->'The cute dog walks in the green park'
```

## Method 2 Using Denoised Training

input=The<extra_id_0> dog walks<extra_id_1> park<extra_id_2></s>
target=<extra_id_0> cute<extra_id_1> in the green<extra_id_2></s></s>

Output
```
len_tokenizer=32100
--------------------
encoded_prompt =tensor([[   37, 32099, 10681,    16, 32098,  2447,     1]])
encoded_labels =tensor([[32099,  5295,  1782, 32098,     8, 32097,     1]])
torch.Size([1, 7]) =torch.Size([1, 7])
--------------------
denoised input_ids decoded = The<extra_id_0> dog walks<extra_id_1> park<extra_id_2></s>
denoised labels decoded   = <extra_id_0> cute<extra_id_1> in the green<extra_id_2></s></s>
input_ids.shape (1, 8) labels.shape (1, 9)
Before Training:'The cute dog walks in the'-->'Als'
Epoch 0  Loss 3.3623902797698975
Epoch 10  Loss 1.6039639711380005
Epoch 20  Loss 1.2029192447662354
Epoch 30  Loss 0.7536050081253052
Epoch 40  Loss 1.022627353668213
Epoch 50  Loss 0.11967131495475769
Epoch 60  Loss 0.08717009425163269
Epoch 70  Loss 0.08708580583333969
Epoch 80  Loss 0.027843305841088295
Epoch 90  Loss 0.05927355960011482
Epoch 99  Loss 0.06003263220191002
After Training:'The cute dog walks in the'-->'cute dog walks in the cute cute dog'
```

# Method 3 Using Target as is

input = "The cute dog walks in the"
target = "green park"

Ouput
```
Before Training:'The cute dog walks in the'-->'Der cute dog walks in the the'
Epoch 0  Loss 9.206318855285645
Epoch 10  Loss 6.024537563323975
Epoch 20  Loss 3.063105821609497
Epoch 30  Loss 3.7331817150115967
Epoch 40  Loss 1.68136465549469
Epoch 50  Loss 0.3739849328994751
Epoch 60  Loss 0.13449843227863312
Epoch 70  Loss 0.09859928488731384
Epoch 80  Loss 0.4800107479095459
Epoch 90  Loss 0.10699515789747238
Epoch 99  Loss 0.02133219689130783
After Training:'The cute dog walks in the'-->'green park'
```

For larger data
## with bigger number denoiser = FlaxDataCollatorForT5MLM(tokenizer_len,eos_token_id,.55,1.5)

```
labels=torch.Size([1, 118]), attention_mask=torch.Size([1, 106])
denoised input_ids decoded = camera<extra_id_0> focus<extra_id_1> during continuous<extra_id_2> <extra_id_3> on<extra_id_4> quality setting<extra_id_5> if<extra_id_6> exposure<extra_id_7> is<extra_id_8> Auto<extra_id_9> <extra_id_10> High ISO speed<extra_id_11> is set<extra_id_12> Function<extra_id_13> the<extra_id_14> shooting<extra_id_15> may<extra_id_16> slower<extra_id_17> the<extra_id_18> burs<extra_id_19> continuous<extra_id_20> may<extra_id_21> <extra_id_22> o<extra_id_23> mode<extra_id_24> continuous shooting<extra_id_25> become<extra_id_26> slower<extra_id_27> on<extra_id_28> subject<extra_id_29> lens used<extra_id_30> continuous<extra_id_31> speed<extra_id_32> indoor<extra_id_33> and<extra_id_34> low<extra_id_35> Using the<extra_id_36> the<extra_id_37> j<extra_id_38> the<extra_id_39> Press the<extra_id_40> or<extra_id_41> the dial<extra_id_42> select<extra_id_43> self<extra_id_44> r<extra_id_45> plus<extra_id_46></s>
denoised labels decoded   = <extra_id_0> will<extra_id_1> only once<extra_id_2> shooting<extra_id_3> Depending<extra_id_4> the<extra_id_5> <extra_id_6> Long<extra_id_7> noise reduction<extra_id_8> set to<extra_id_9> or On or<extra_id_10> if<extra_id_11> noise reduction<extra_id_12> to Strong under Custom<extra_id_13> s<extra_id_14> continuous<extra_id_15> speed<extra_id_16> be<extra_id_17> and<extra_id_18> maximum<extra_id_19> t during<extra_id_20> shooting<extra_id_21> decrease In AI<extra_id_22> Serv<extra_id_23> AF<extra_id_24> the<extra_id_25> speed may<extra_id_26> slightly<extra_id_27> depending<extra_id_28> the<extra_id_29> and the<extra_id_30> The<extra_id_31> shooting<extra_id_32> might also decrease<extra_id_33> s<extra_id_34> under<extra_id_35> light <extra_id_36> Press<extra_id_37> Yi<extra_id_38> button<extra_id_39> Select<extra_id_40> key<extra_id_41> turn<extra_id_42> to<extra_id_43> the desired<extra_id_44> time<extra_id_45> then press<extra_id_46> continuous shots Press the</s>
Loss
tensor(9.0541, device='cuda:0', grad_fn=<NllLossBackward0>)
```

## with smaller number denoiser = FlaxDataCollatorForT5MLM(tokenizer_len,eos_token_id,.15,1.5)

```
tensor(4.4986, device='cuda:0', grad_fn=<NllLossBackward0>)
----------------------------------------------------
denoised input_ids decoded = information Turn the camera off and on<extra_id_0> if<extra_id_1> is set to Disable it may still transmit signal In hospitals airports and other<extra_id_2> wireless transmissions are prohibited remove the card from the camera If the image transfer does not function<extra_id_3> the card and personal<extra_id_4> settings For details see the card<extra_id_5> <extra_id_6></s>
denoised labels decoded   = <extra_id_0> again Even<extra_id_1> trans<extra_id_2> places where<extra_id_3> check<extra_id_4> computer<extra_id_5> instruction manual<extra_id_6> Depending</s>
----------------------------------------------------
denoised input_ids decoded = copyright information it will be appended to the image as<extra_id_0> if<extra_id_1> Copyright information Under<extra_id_2> tab select Copyright information then press the option to be set<extra_id_3> Enter author name or Enter copyright details then press The text entry screen will appear<extra_id_4> Display copyright info to check the<extra_id_5> right information<extra_id_6></s>
denoised labels decoded   = <extra_id_0> Ex<extra_id_1> information Select<extra_id_2> the<extra_id_3> Select Select<extra_id_4> Select<extra_id_5> copy<extra_id_6> currently set</s>
```