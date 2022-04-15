# -*- mode: python ; coding: utf-8 -*-


block_cipher = None
name_list=['CarMain.py','CTPN_detect.py','HAAR_detect.py','YOLO_detect.py',
			'nets/model_train.py','nets/vgg.py',
			'utils/bbox/bbox_transform.py','utils/bbox/setup.py',
			'utils/dataset/data_provider.py','utils/dataset/data_util.py',
			'utils/prepare/split_label.py','utils/prepare/utils.py',
			'utils/rpn_msr/anchor_target_layer.py','utils/rpn_msr/config.py',
			'utils/rpn_msr/generate_anchors.py','utils/rpn_msr/proposal_layer.py',
			'utils/text_connector/detectors.py','utils/text_connector/other.py',
			'utils/text_connector/text_connect_cfg.py','utils/text_connector/text_proposal_connector.py',
			'utils/text_connector/text_proposal_connector_oriented.py','utils/text_connector/text_proposal_graph_builder.py']

a = Analysis(name_list,
             pathex=[],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts, 
          [],
          exclude_binaries=True,
          name='CarMain',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas, 
               strip=False,
               upx=True,
               upx_exclude=[],
               name='CarMain')
