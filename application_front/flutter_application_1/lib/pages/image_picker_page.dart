import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:permission_handler/permission_handler.dart';
import 'diagnosis_result_page.dart';
import 'image_crop_page.dart';
import 'diagnosis_result_page.dart';

class ImagePickerPage extends StatefulWidget {
  @override
  State<ImagePickerPage> createState() => _ImagePickerPageState();
}

class _ImagePickerPageState extends State<ImagePickerPage> {
  final ImagePicker _picker = ImagePicker();
  File? _image;

  Future<void> _requestPermissions() async {
    await [Permission.camera, Permission.photos, Permission.storage].request();
  }

  Future<void> _getImage(ImageSource source) async {
    await _requestPermissions();

    final XFile? picked = await _picker.pickImage(source: source);
    if (picked != null) {
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => ImageCropPage(imagePath: picked.path),
        ),
      );
    }
  }

  Future<void> _uploadImage(File imageFile) async {
    final uri = Uri.parse(
      "http://192.168.10.17:8000/predict",
    ); // //////////////////////////이부분 바꿔서 실행
    final request = http.MultipartRequest('POST', uri)
      ..files.add(await http.MultipartFile.fromPath('file', imageFile.path));

    final response = await request.send();

    if (response.statusCode == 200) {
      final respStr = await response.stream.bytesToString();
      final json = jsonDecode(respStr);

      Navigator.push(
        context,

        MaterialPageRoute(
          builder: (context) => DiagnosisResultPage(result: json),
        ),
      );
    } else {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('진단 실패: 서버 오류')));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("타이어 진단")),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton.icon(
              icon: Icon(Icons.camera_alt),
              label: Text("카메라로 촬영"),
              onPressed: () => _getImage(ImageSource.camera),
            ),
            SizedBox(height: 20),
            ElevatedButton.icon(
              icon: Icon(Icons.photo_library),
              label: Text("갤러리에서 선택"),
              onPressed: () => _getImage(ImageSource.gallery),
            ),
          ],
        ),
      ),
    );
  }
}
