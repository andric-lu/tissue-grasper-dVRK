# psm_Si_model/ — where this asset came from

A dVRK PSM (Si variant) URDF plus its mesh assets, added directly to the
repository rather than fetched at build time — mirrors the convention in
`third_party/PROVENANCE.md`, which this file follows.

## TODO — source and license (not yet filled in)

**This section is incomplete and blocks committing this directory.** The
URDF's own header comments reference `$(find dvrk_model)/model/Si/psm_si.xacro`
and a personal development path
(`/home/yilincai/dvrk_sim_ws/src/ros2_dvrk_model/model/Si/tools.urdf.xacro`),
suggesting origin in a `ros2_dvrk_model`-style ROS package — this is inferred
from the file's own text, not confirmed against an actual upstream source.

Needed before this is committed:

| | |
|---|---|
| Upstream repository / URL | **TODO** |
| Commit / release, if known | **TODO** |
| License | **TODO** — must be identified before this tree is redistributed in this repository's git history |
| Retrieved on | 2026-08-31 |
| Retrieved by | Andric (added directly to the working tree, not via `git clone`) |

## What's actually used by this project

Only `psm_si_surrol.urdf` and the 13 mesh files it references are used by
`host/psm.py`:

- `meshes/Si/PSM_ECM/link_{0,1,2,3,4}.STL` (5 files)
- `meshes/instruments/420006/{tool_main_link, tool_wrist_link,
  tool_wrist_scal_link, tool_wrist_sca_shaft_link,
  tool_wrist_sca_ee_link_{1,2}, tool_wrist_shaft_link}.STL` (8 files)

The rest of `meshes/` (Classic dVRK PSM/ECM/MTM/SUJ assets, alternate
instrument tips, `tower.STL`) arrived as part of the same source tree and is
kept for provenance/future use, but nothing in this repository currently
reads it.

## Files, as added

`sha256sum` over every file in the tree (122 files, `.DS_Store` excluded —
already covered by the repo's own `.gitignore`), so a later "did I change
this?" is answerable without the network. Generated with:

```bash
find psm_Si_model -type f -exec shasum -a 256 {} \; | sort -k2
```

```
2bb0f03e59eb1b0aed76695457edeae3cdc7921e891fd055508ba4bf83e0327d  meshes/Classic/dVRK-cube.stl
79aec0fbc8c1dd9a2e22f59320b4512ab704d32db2b4980b7fed51d9528ddd94  meshes/Classic/ecm/ecm_base_link.STL
aa36a4d40178e330b60d382b54e657870efd5a5b9c54f77a5ae72edf82601558  meshes/Classic/ecm/ecm_insertion_link.STL
b5b1024ee6619406e7a39e94b6a4f6dd92cbc9237cd093294a940eab04ec2a82  meshes/Classic/ecm/ecm_pitch_link_1.STL
b415418b9a8a51b958e23d096f5a5e91b8d5d4ac6d2312cbeb19cd293ee93ade  meshes/Classic/ecm/ecm_pitch_link_2.STL
a8cf41dadee85464c730d5a7110b7903f2163488db4e4dd25e9b8f0282fac0bc  meshes/Classic/ecm/ecm_pitch_link_3.STL
149b8b6a849996361095bed85183570143cb7ab66c7a94998f2504b2a39cf64e  meshes/Classic/ecm/ecm_roll_link.STL
3dfbf14f45a84f48423d0c86145b8690d52696de791f0e3d549ffb02234ce821  meshes/Classic/ecm/ecm_yaw_link.STL
d684df60dfee5c1ccbe9d6d0a48faf5c06bfc310dc91324f1acb2a963f3ed533  meshes/Classic/ecm/Endo_Arm.STL
a06da1100cdec4646f8c8a70325bc93c492c3fe54485c6790129766a744b0a4e  meshes/Classic/ecm/Endo_Link_5_2.STL
385a332836a4a3932142605d6e375c3894754d26dc5a09c0c8b2fc1dc2805b3c  meshes/Classic/ecm/Endo_Link_5.STL
af66f615ca26c15dc270232bcea37f5bd79a73e65d107ca89c4d109e847a13ad  meshes/Classic/ecm/Endo_Link_6_2.STL
9cfa8d479b85014dcfa17ed1131c8d062e41a6de5c4343a9a2d61dd4948a0597  meshes/Classic/ecm/Endo_Link_7.STL
e06c2347e155ab2831c418ac4a198485712dd68bb2169bf66f41fa3abf1bbd50  meshes/Classic/ecm/Endo_Link_8.STL
0c0ec3b6ea86048c5901f88752931465d3ce82980660d5003ca6f4eb61217509  meshes/Classic/ecm/EndoScope.STL
2023e37a2cede51a4c9db34dad220ca8eac7e28382fc10cc59e4b7bfa99936a9  meshes/Classic/mtm/ArmParallel.dae
f8674e63d38c88dc34ecf363fe2fdf43bc12e1e30a76a92965a10856de4282a5  meshes/Classic/mtm/ArmParallel.STL
814c76ae5c9043c8c7d3e876cf0a2480e7e18d52f9029111d5aa97ddd75373dc  meshes/Classic/mtm/ArmParallel1.dae
e42691cf617483a4c69fbf6596ae7c876d9c20b46090b6de8dc4674c00191be8  meshes/Classic/mtm/ArmParallel1.STL
d0b2e73eb000b03b7fa555573ea964bda20cbfa2596f6f216782057ca0c62a72  meshes/Classic/mtm/BottomArm.dae
81e31164180d04e83339eb43f75b0dc5cb12540965494d10d6eb26fad206bb0e  meshes/Classic/mtm/BottomArm.STL
be978e53d8a0d94d7064da664b8bfb7130b75f5958173b1047072aa5e6466a32  meshes/Classic/mtm/Link.dae
d4bc96548d03558a1b1bd63eb9fa7fdec530a814938f9a897406f858100e843e  meshes/Classic/mtm/Link.STL
e7b29d9039cff20e9ef23bd1417ace4724fd2e0a160363dda0786c2828888b88  meshes/Classic/mtm/MTM_model.urdf
788925ceabe94040ded580a4e6ab21b6585df9131ed3fc247990ee053ce1987a  meshes/Classic/mtm/mtm_omponents.blend
d7bafcb76d7753b9fd9cb5e519990528db644ae4de6d999f14d5a6fbec8a2a19  meshes/Classic/mtm/mtm_wrist_pitch.png
011a09f56d8cba2e7e3b05f091473eb31dd2bd5b68be7eead9f1c8e4420dab1c  meshes/Classic/mtm/mtm_wrist_platform.png
1423fad6c8635ef6618dbc5facaa10468c13791237b85c9b2826c8dd8262c811  meshes/Classic/mtm/mtm_wrist_yaw.png
68ff661104f57b9487d691eff8e6bbf816f85e0b10cb6da1e82ab71e458cc3e8  meshes/Classic/mtm/OutPitch_Shoulder.dae
69b38a09d49a9062afffb74f7786a962de82ff5f03e7a5d6cb4b9ce7430fb7c5  meshes/Classic/mtm/OutPitch_Shoulder.STL
9b679c16fa749e3d7975b73c704ac5e41b171a759c6058c569202141f4d84363  meshes/Classic/mtm/Top Panel.dae
dab4cab13ee36a4a02aa8dd4ef8453802207192c6eb2edf60644515f75059f94  meshes/Classic/mtm/Top Panel.STL
139308790ef62a9a1318935cb04fdb8352b10121e992077664dfe0f5db4d8f04  meshes/Classic/mtm/WristPitch.dae
d9408e4c4b8cfecc070678185150f8d13b3d0db4eefb7d05e327035a5ede0449  meshes/Classic/mtm/WristPitch.STL
6cabc55e79b0d5deae3ae88e6aff1bf353b9e72fc3cc2ff93a344f60bf473b9e  meshes/Classic/mtm/WristPlatform.dae
8037580bf93671d6fdd3bcd9d773fd7a71fc22d313199ccd93afc9d43eee49d2  meshes/Classic/mtm/WristPlatform.STL
b05a9a866420fa13e4bcf7236e0442ce49818c4f217416d431de83bd210231a3  meshes/Classic/mtm/WristRoll.dae
b2c79730056c1c5da0be6ef7c206ce2c509842d5e21e65b850f3684f68f0ec13  meshes/Classic/mtm/WristRoll.STL
960357ad3bdac1832967fe21b7f8fb9b0b86910aacb414020963c9b087f298f8  meshes/Classic/mtm/WristYaw.dae
4f1b1a3bdc18b1ee6834ebbecddc73ba74c94661b69fc9f8a1ad6fb885957558  meshes/Classic/mtm/WristYaw.STL
a289af1f0fb6ad50a4964146326235cd1e98ee68cdec1671d692568779157512  meshes/Classic/psm/knife.STL
a13f2e1f841380766a74594620b6e7a23af907c8b2dde1397441a3a81e90efdd  meshes/Classic/psm/outer_insertion.dae
017b8b710341ff345069123a05e6a92237cd0d962eb69ad12b651b67deead996  meshes/Classic/psm/outer_insertion.stl
e734f45299e5354647eb35980d7cc4921c7c1de8f439516c70634a9e7765efb1  meshes/Classic/psm/outer_pitch_back.dae
809aea44d2eb6bac4c70388b8f5ed8d881fef48923cdfe4b7f15dc92e928d50d  meshes/Classic/psm/outer_pitch_back.stl
5d2e5e7346487001d3ab1c26b111d4314e97b7219d1e96bfe7650a824bbe3ef5  meshes/Classic/psm/outer_pitch_bottom.dae
0450391c799ff7b351d21b163a4860bfe36387dca5aed5e5503a891e4e29ebdd  meshes/Classic/psm/outer_pitch_bottom.stl
b5c19b7dfb03d2e985d81f972e1619bab562af81c2d450b0e45edb1dcb72653d  meshes/Classic/psm/outer_pitch_front.dae
922b6573576f92fe5f94a87edf538d024edb7c4a3be8beae5e87a92418cb8c1f  meshes/Classic/psm/outer_pitch_front.stl
317b4f7b2b2307d42c0fb6b616464173fa2207cd05898fabded4a9d9c993a6c7  meshes/Classic/psm/outer_pitch_top.dae
b717c9e61c6517a2f007a9e6ee4bc55a1e0d02d0cdde6e28dfe52028e4f17a94  meshes/Classic/psm/outer_pitch_top.stl
3061f150eedeb65a7d5dd0a93df45ee49eb4905597aa705ab3cfe96e60cfa0b9  meshes/Classic/psm/outer_yaw.dae
3a526f656e487ed36c9e77e23e914108321c1906909bfd549d883b476cee9fcd  meshes/Classic/psm/outer_yaw.stl
fb06e0c31466b69d286970e6d850a203b7e875896cc9a9bb70e9b377b6ab71a9  meshes/Classic/psm/psm_base.dae
34fd536aea1b9970ff89bd48d5de7094d17d2ae245c8b16eaec36748f03c8a92  meshes/Classic/psm/psm_base.stl
bd940fc71fe8fcb4010632004ce551416aae3e0fce43fe8f6a6dd3743d03e59c  meshes/Classic/psm/snake_tool/gripper_2.STL
bd940fc71fe8fcb4010632004ce551416aae3e0fce43fe8f6a6dd3743d03e59c  meshes/Classic/psm/snake_tool/gripper_3.STL
f24a761d648af01dd2d1bc2e4d9a78c149ece5b531a6ad589f0c821fb9867f5d  meshes/Classic/psm/snake_tool/link_0.STL
6cc2d25ddac19ea502a1b0e9b35f075af5818f8c560d98fa99b3c9a775173987  meshes/Classic/psm/snake_tool/link_1.STL
b8f50bf4a3e633c7fb4ac377cac2a585758d12d380dd09a5da15cb73f1408e07  meshes/Classic/psm/snake_tool/link_2.STL
c8a1faf9d9338da6179c72ed95e105288ba9f75e4ef11054a95fbd2d9cccc603  meshes/Classic/psm/snake_tool/link_3.STL
ba0ea5e333a47ab194be984e6180d065a646c582534ad1b8c6ddca01d036817d  meshes/Classic/psm/snake_tool/link_4.STL
fa5b2a3893d7000623b01c798d81d0b32a4e48ed4b2abb18f5ed06c704bd3a58  meshes/Classic/psm/tool_adapter.dae
3c31c8a6e5bd5707d23d1c459c68953e84752593be2c7e9999007a167ff62875  meshes/Classic/psm/tool_adapter.stl
4e67208d7eb3b458363aad2f50226de67f3f892580e03a1138abb8a6b774bcec  meshes/Classic/psm/tool_main.dae
42a664f6bda13c264694c9d393701e1ce6297f063a8fef15338be8c48658ff47  meshes/Classic/psm/tool_main.stl
5158eef993caade1ae79fe4f05331a43b68992fea35f14bb839b0f2b99200530  meshes/Classic/psm/tool_wrist_caudier_link_1_shaft.stl
e74d26ebf66138b24f1d730998c4df407ae94d6512b03aeaa1f474f3a42e7aeb  meshes/Classic/psm/tool_wrist_caudier_link_1.stl
a5206c7d510f38229771366a285cb56fbaf5ede753c20fd0ce77f8acdcada270  meshes/Classic/psm/tool_wrist_caudier_link_2.stl
7f10df546cf600d822ef10f2ec4976357c3547671bcfccec77c8b8a92892bce1  meshes/Classic/psm/tool_wrist_link.dae
388635dc798690db0427fc679383f6475f5d5af8f37c51d30bf631b962ba4fa8  meshes/Classic/psm/tool_wrist_link.stl
59e214c0fea4c27986681757af8babb02cbed6aa0ab5926850175c6dfc836435  meshes/Classic/psm/tool_wrist_sca_link_2.dae
8a986d8c234fa26c891ad18c76a84e27c0e2cfd966b12506670d85e179099d60  meshes/Classic/psm/tool_wrist_sca_link_2.stl
cae88e68c8322e67c24a5518fa25a834922f23ba7d1723c78184a8a36acdeb76  meshes/Classic/psm/tool_wrist_sca_link.dae
27094da2e1d3b6ed71a77688746febc22adf7d2bc9f572a9cde9630e0e270ad8  meshes/Classic/psm/tool_wrist_sca_link.stl
4f22227c53789ebddbc938db857918adcd2b42a945fe650afa3b3024fe236ae6  meshes/Classic/psm/tool_wrist_sca_shaft_link.dae
6746e2677de1dac89a190640ad6cbebd2f393918dd94b55a88ffce958695b28c  meshes/Classic/psm/tool_wrist_sca_shaft_link.stl
5ce7f996e3be6019bc2b319cb91a324615dd6831540851b72eb1cc9b7a0da993  meshes/Classic/psm/tool_wrist_shaft_link.dae
c4bff19ed21230aaa70ab1dd9c5838bf58b31335f03e5b1b0403d2e2f397318d  meshes/Classic/psm/tool_wrist_shaft_link.stl
149bfe65ba84c1f0418816aea5210a89bf838db00675c68af9c5080900d97682  meshes/instruments/420006/tool_main_link.STL
720c4517dad258641ae306c1c5d075ae1614dfb0ca247ffd177ce51b9ac3be15  meshes/instruments/420006/tool_wrist_link.STL
f9403caa6b059807d045896c4de21bd1a46e9c48f7a3cfe1b5b906980a58e729  meshes/instruments/420006/tool_wrist_sca_ee_link_1.STL
36cd8f3fcf3fe581d34dbae6980e6518ea3793de228679d534076df438c09835  meshes/instruments/420006/tool_wrist_sca_ee_link_2.STL
b515f1c01e9b8a5684b88c44c1210e6eae5034f8926817fdc7e496d1cafa4804  meshes/instruments/420006/tool_wrist_sca_link.STL
0fa15767709f121d3482d42329027404c897a29ec1b3aff967c2c6a66501707b  meshes/instruments/420006/tool_wrist_sca_shaft_link.STL
a49ae6007dcd2f4aed7e620df3ffd27b40afef48aafba6a39182d9537d8eab33  meshes/instruments/420006/tool_wrist_scal_link.STL
b8bd34660bec341978f42c38d212d3cd04f91e51f9e0d6a37ff3b797ab07be7b  meshes/instruments/420006/tool_wrist_shaft_link.STL
d5f5002bd9afd9817eb95526a04b78c5ff99f6899b61e6ff92f68b2a3e4700e8  meshes/instruments/SF0826001/tool_main_link.STL
5fb3ec976a2796705a937eccdce23f8cc2219884b96e6674d233b43fd17733a8  meshes/instruments/SF0826001/tool_roll_link.STL
f68cab28dbd43c625e5618a5bb298f81d2a7df3e6214ebfc1e02a533c74ded78  meshes/instruments/tool/main_insertion_link_1.obj
ae19094bf6ef1d096bb57f6f73d0729a3e40033349c97c194ea66f40b4f17206  meshes/instruments/tool/main_insertion_link_2.obj
0d55950ec692223a8615b247da6ec0ce3fcdc487b9c07c1d24eab9d919280d84  meshes/instruments/tool/main_insertion_link_3.obj
f4193ad1de717e4d3438af4f78d77653a3e954043ca9d67e84fd5f9ab871545b  meshes/instruments/tool/main_insertion_link.STL
a661c61ec9354cbdeb04e336031784ee131c7bd9764c621bee1d3b6d885a3db5  meshes/instruments/tool/tool_gripper1_link_wide.obj
b647321b30143ca7015120bda1e163f3fa42e57cda20981acc5ab7f47c0c4155  meshes/instruments/tool/tool_gripper1_link.obj
605968ecfae853f80a690ec0c2449890168d383d5f4092ac8f4ebebf8583c7ac  meshes/instruments/tool/tool_gripper1_link.STL
899a3264c40f9069ab13cbefd92e85f9750a6bf014701267db5a8bcad3862f03  meshes/instruments/tool/tool_gripper2_link_wide.obj
8e49203fe99fa78beb6e49efbf8aba3ca21c99e90326e857402a12cf0ff6f0dd  meshes/instruments/tool/tool_gripper2_link.obj
8f4b041d872a15520ae095768f9c95e54b85469ebff9cda1698593199bc5d7a3  meshes/instruments/tool/tool_gripper2_link.STL
f067901e324a84ceb0d3bda498a5c8d5bd977c741ef9111e81ef143e7d2629b1  meshes/instruments/tool/tool_pitch_link.STL
5f847b435ebc501345ae7c6fa31edee811ab3008d1b804849e12d3c94f3f8107  meshes/instruments/tool/tool_roll_link.STL
09bc3484135e551932042a270a4bb71b4ccd08656f13c61f74bb16ba1ea1aa1a  meshes/instruments/tool/tool_yaw_link.STL
d3922b0db21b524977494ae4c837695e335a57aa31d0ca11e16582fb8ee34d26  meshes/Si/PSM_ECM/link_0.STL
6f543ee4ee81db186f4cd841315fd13d54a4063b851fe50440e1ae5e30f1ee4f  meshes/Si/PSM_ECM/link_1.STL
c835f1eddf9ca2d604b390f0fe29e4eead8f3eeb6ac787a4a7679c7939d41714  meshes/Si/PSM_ECM/link_2.STL
ee413eff48045c71ce192e6f7f7859d8979830a44ed79ab167672d9e70011681  meshes/Si/PSM_ECM/link_3.STL
a8a41ac14fae4c23d3ac07eb053de120a14c6ed7aa675d7faa269fd388a0aead  meshes/Si/PSM_ECM/link_4.STL
8bf98d88a7a115161c081ee24cea30a2672276410652c766d32e4c8846bee20e  meshes/Si/SUJ/ECM/link_0.STL
85a9fd2ac7cb73ad73ee5992a57110c899d492b849bcd642fe46abd0ca94a500  meshes/Si/SUJ/ECM/link_1.STL
669e28b1f17fb159cce4f343c402fb7cfe2a024b86564c77c597eadfa607cfcd  meshes/Si/SUJ/ECM/link_2.STL
240251a1c21a6071c958ad4ce54421eaabb05384975b0b07b8a4a6a80ce5243d  meshes/Si/SUJ/ECM/link_3.STL
c7a106a2adb1214336caac238b66ebf37f051b4b8c29976fa72923bde5c8b683  meshes/Si/SUJ/PSM/12/link_0.STL
779c791e66a76b8eea9e07d431c2fb46e207a30d6cfafe75d19359e42ad02c9e  meshes/Si/SUJ/PSM/12/link_1.STL
085dc5933db1351a1251e1294c08143ce024741d851061dc4429adcc2958ddb6  meshes/Si/SUJ/PSM/12/link_2.STL
1aaa5558a6a793870f5379befadac4db760284acc3665697e3c6a898e1bdc39e  meshes/Si/SUJ/PSM/12/link_3.STL
da83df578bb812f2e78451a91153800497f8754ae42fec8225ff5099fb678f79  meshes/Si/SUJ/PSM/3/link_0.STL
9ae3bd7fff4acea428c8685440bb1509a45d98122dbb9aa5430aeb29e47a9df2  meshes/Si/SUJ/PSM/3/link_1.STL
0e7633b3e0b0a18dcebfd22ac14eca57c2263341c8b3adbb2d5afe2177e2eb88  meshes/Si/SUJ/PSM/3/link_2.STL
085dc5933db1351a1251e1294c08143ce024741d851061dc4429adcc2958ddb6  meshes/Si/SUJ/PSM/3/link_3.STL
a596fd83d37db68e06efee54200f4f2804a9affbafe337532994c27c3bbec35b  meshes/Si/SUJ/PSM/3/link_4.STL
f704dc22c0d614f1b2996e3e456149e3605a456d0013a54885859e6c70ecbfd5  meshes/Si/tower.STL
88b440db2fc2b65b7896504bf773da713a300ab1e9b1305b2ef051dfa466ec37  psm_si_surrol.urdf
```

## Local modifications

**None.** Added as received; not edited.

## Kinematic note (DECISION_LOG.md, PSM integration section)

The URDF's `<mimic>` joints (`j3` mimics `j2`, `j4` mimics `j2`,
`jaw_joint_2` mimics `jaw_joint_1`) are not enforced by PyBullet — it loads
them as independent actuatable joints. `link_2`/`link_3` (the `j2`/`j3`
branch) are the dVRK's decorative parallelogram linkage and are **not**
ancestors of `tool_gripper_center` in the kinematic tree; only
`jaw_joint_2 = -jaw_joint_1` has any effect on tool pose or collision and is
enforced by hand in `host/psm.py`.
