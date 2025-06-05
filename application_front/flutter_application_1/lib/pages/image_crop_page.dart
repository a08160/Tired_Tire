import 'dart:io';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:image_cropper/image_cropper.dart';
import 'package:http/http.dart' as http;
import 'diagnosis_result_page.dart';

class ImageCropPage extends StatefulWidget {
  final String imagePath;

  const ImageCropPage({required this.imagePath});

  @override
  State<ImageCropPage> createState() => _ImageCropPageState();
}

class _ImageCropPageState extends State<ImageCropPage> {
  CroppedFile? _croppedFile;

  Future<void> _cropImageAndUpload() async {
    final cropped = await ImageCropper().cropImage(
      sourcePath: widget.imagePath,
      aspectRatio: const CropAspectRatio(ratioX: 1, ratioY: 1),
      uiSettings: [
        AndroidUiSettings(
          toolbarTitle: '이미지 자르기',
          lockAspectRatio: true,
          initAspectRatio: CropAspectRatioPreset.square,
        ),
        IOSUiSettings(title: '이미지 자르기'),
      ],
    );

    if (cropped != null) {
      setState(() {
        _croppedFile = cropped;
      });

      // 서버로 업로드
      final uri = Uri.parse("http://192.168.10.17:8000/predict"); // ← IP 확인
      final request = http.MultipartRequest('POST', uri)
        ..files.add(await http.MultipartFile.fromPath('file', cropped.path));

      final response = await request.send();

      if (response.statusCode == 200) {
        final respStr = await response.stream.bytesToString();
        final jsonResult = jsonDecode(respStr);
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => DiagnosisResultPage(result: jsonResult),
          ),
        );
      } else {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text("진단 실패: 서버 오류")));
      }
    } else {
      Navigator.pop(context); // 자르기 취소 시 이전 화면으로 복귀
    }
  }

  @override
  void initState() {
    super.initState();
    _cropImageAndUpload();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('이미지 자르기')),
      body: Center(
        child:
            _croppedFile != null
                ? Image.file(File(_croppedFile!.path))
                : CircularProgressIndicator(),
      ),
    );
  }
}
