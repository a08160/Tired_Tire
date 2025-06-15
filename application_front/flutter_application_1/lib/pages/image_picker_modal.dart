import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:permission_handler/permission_handler.dart';
import 'image_crop_page.dart';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'diagnosis_result_page.dart';

class ImagePickerModal extends StatefulWidget {
  @override
  State<ImagePickerModal> createState() => _ImagePickerModalState();
}

class _ImagePickerModalState extends State<ImagePickerModal> {
  final ImagePicker _picker = ImagePicker();

  Future<void> _requestPermissions() async {
    await [Permission.camera, Permission.photos, Permission.storage].request();
  }

  Future<void> _getImage(ImageSource source) async {
    await _requestPermissions();

    final XFile? picked = await _picker.pickImage(source: source);
    if (picked != null) {
      await _uploadImage(picked.path);
    }
  }

  Future<void> _uploadImage(String imagePath) async {
    // ✅ 로딩 다이얼로그 띄우기
    showDialog(
      context: context,
      barrierDismissible: false, // 바깥 터치시 닫히지 않게
      builder: (context) {
        return Center(
          child: Container(
            width: 120,
            height: 120,
            decoration: BoxDecoration(
              color: Colors.black87,
              borderRadius: BorderRadius.circular(16),
            ),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                CircularProgressIndicator(color: Colors.white),
                SizedBox(height: 20),
                Text("진단 중...", style: TextStyle(color: Colors.white)),
              ],
            ),
          ),
        );
      },
    );

    try {
      final uri = Uri.parse("http://192.168.0.25:8000/predict");
      final request = http.MultipartRequest('POST', uri)
        ..files.add(await http.MultipartFile.fromPath('file', imagePath));

      final response = await request.send();

      if (response.statusCode == 200) {
        final respStr = await response.stream.bytesToString();
        final jsonResult = jsonDecode(respStr);
        if (!mounted) return;

        Navigator.pop(context); // ✅ 로딩창 닫기
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => DiagnosisResultPage(result: jsonResult),
          ),
        );
      } else {
        Navigator.pop(context); // ✅ 로딩창 닫기
        ScaffoldMessenger.of(
          Navigator.of(context, rootNavigator: true).context,
        ).showSnackBar(SnackBar(content: Text("진단 실패: 서버 오류")));
      }
    } catch (e) {
      Navigator.pop(context); // ✅ 로딩창 닫기
      ScaffoldMessenger.of(
        Navigator.of(context, rootNavigator: true).context,
      ).showSnackBar(SnackBar(content: Text("네트워크 오류: $e")));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.vertical(top: Radius.circular(30)),
      ),
      // 이 부분이 모달의 높이를 거의 전체로 키우는 핵심입니다!
      constraints: BoxConstraints(
        maxHeight: MediaQuery.of(context).size.height * 0.9,
      ),
      child: SingleChildScrollView(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            /// ✅ 상단 라벨
            Container(
              padding: EdgeInsets.symmetric(horizontal: 16, vertical: 6),
              decoration: BoxDecoration(
                color: Color(0xFFEDF4FF),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text(
                '타이어 공기압 진단',
                style: TextStyle(
                  color: Color(0xFF1C3FAA),
                  fontWeight: FontWeight.bold,
                  fontSize: 14,
                ),
              ),
            ),

            SizedBox(height: 20),

            /// ✅ 안내 문구
            Text(
              '타이어의 옆면 전체를\n정면으로 찍어주세요',
              textAlign: TextAlign.center,
              style: TextStyle(fontWeight: FontWeight.bold, fontSize: 20),
            ),

            SizedBox(height: 20),

            /// ✅ 예시 이미지 (Good / Bad)
            _buildExampleImages(),

            SizedBox(height: 30),

            /// ✅ 촬영 / 갤러리 선택 버튼
            _buildSelectButtons(),

            SizedBox(height: 30),

            /// ✅ 하단 주의사항
            Text(
              '주의 사항\n• 본 진단은 AI로 진행하는 간이 검사이므로\n  정확한 공기압 정도와는 다소 차이가 있을 수 있습니다.',
              style: TextStyle(color: Colors.grey[600], fontSize: 12),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }

  /// ✅ Good / Bad 예시 이미지 부분 따로 함수로 분리
  Widget _buildExampleImages() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        _buildExampleItem(
          imagePath: 'assets/tire_good_example.png',
          label: 'Good',
          labelColor: Colors.green,
        ),
        SizedBox(width: 20),
        _buildExampleItem(
          imagePath: 'assets/tire_bad_example.png',
          label: 'Bad',
          labelColor: Colors.red,
        ),
      ],
    );
  }

  /// ✅ 각각의 예시 이미지 박스 구성
  Widget _buildExampleItem({
    required String imagePath,
    required String label,
    required Color labelColor,
  }) {
    return Column(
      children: [
        ClipRRect(
          borderRadius: BorderRadius.circular(16),
          child: Image.asset(
            imagePath,
            width: 140,
            height: 140,
            fit: BoxFit.cover,
          ),
        ),
        SizedBox(height: 8),
        Container(
          padding: EdgeInsets.symmetric(horizontal: 12, vertical: 4),
          decoration: BoxDecoration(
            color: labelColor,
            borderRadius: BorderRadius.circular(12),
          ),
          child: Text(
            label,
            style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
          ),
        ),
      ],
    );
  }

  /// ✅ 선택 버튼 구성
  Widget _buildSelectButtons() {
    return Column(
      children: [
        ElevatedButton.icon(
          onPressed: () => _getImage(ImageSource.camera),
          icon: Icon(Icons.camera_alt),
          label: Text("카메라로 촬영"),
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.black,
            foregroundColor: Colors.white,
            minimumSize: Size(double.infinity, 50),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(30),
            ),
          ),
        ),
        SizedBox(height: 16),
        ElevatedButton.icon(
          onPressed: () => _getImage(ImageSource.gallery),
          icon: Icon(Icons.photo_library),
          label: Text("갤러리에서 선택"),
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.black,
            foregroundColor: Colors.white,
            minimumSize: Size(double.infinity, 50),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(30),
            ),
          ),
        ),
      ],
    );
  }
}
