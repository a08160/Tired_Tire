import 'dart:io';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:http/http.dart' as http;
import 'crack_result_page.dart';

class CrackImageModal extends StatefulWidget {
  final String userName; // ✅ 추가

  CrackImageModal({required this.userName});

  @override
  State<CrackImageModal> createState() => _CrackImageModalState();
}

class _CrackImageModalState extends State<CrackImageModal> {
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
    _showLoading();

    try {
      final uri = Uri.parse("http://192.168.10.11:8001/crack");
      final request = http.MultipartRequest('POST', uri)
        ..files.add(await http.MultipartFile.fromPath('file', imagePath));

      final response = await request.send();

      if (response.statusCode == 200) {
        final respStr = await response.stream.bytesToString();
        final jsonResult = jsonDecode(respStr);
        if (!mounted) return;

        Navigator.pop(context); // 로딩창 닫기
        Navigator.push(
          context,
          MaterialPageRoute(
            builder:
                (context) => CrackResultPage(
                  result: jsonResult,
                  userName: widget.userName,
                ),
          ),
        );
      } else {
        Navigator.pop(context);
        _showTopMessage("타이어를 인식할 수 없습니다. 다시 시도해주세요");
      }
    } catch (e) {
      Navigator.pop(context);
      _showTopMessage("네트워크 오류: $e");
    }
  }

  void _showLoading() {
    showDialog(
      context: context,
      barrierDismissible: false,
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
                SizedBox(height: 20), // 원래 20이었음
                Text("진단 중...", style: TextStyle(color: Colors.white)),
              ],
            ),
          ),
        );
      },
    );
  }

  // 👉 핵심 Overlay 메시지 함수
  void _showTopMessage(String message) {
    final overlay = Overlay.of(context);
    final overlayEntry = OverlayEntry(
      builder:
          (context) => Positioned(
            top: 50,
            left: 20,
            right: 20,
            child: Material(
              color: Colors.transparent,
              child: Container(
                padding: EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                decoration: BoxDecoration(
                  color: Colors.black87,
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Text(
                  message,
                  style: TextStyle(color: Colors.white, fontSize: 16),
                  textAlign: TextAlign.center,
                ),
              ),
            ),
          ),
    );

    overlay.insert(overlayEntry);
    Future.delayed(Duration(seconds: 2), () {
      overlayEntry.remove();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.vertical(top: Radius.circular(30)),
      ),
      constraints: BoxConstraints(
        maxHeight: MediaQuery.of(context).size.height * 0.9,
      ),
      child: SingleChildScrollView(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            /// 상단 라벨
            Container(
              padding: EdgeInsets.symmetric(horizontal: 16, vertical: 6),
              decoration: BoxDecoration(
                color: Color(0xFFEDF4FF),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text(
                '타이어 균열 진단',
                style: TextStyle(
                  color: Color(0xFF1C3FAA),
                  fontWeight: FontWeight.bold,
                  fontSize: 14,
                ),
              ),
            ),
            SizedBox(height: 20),

            /// 안내문구
            Text(
              '진단할 부분을 확대해\n찍어주세요',
              textAlign: TextAlign.center,
              style: TextStyle(fontWeight: FontWeight.bold, fontSize: 20),
            ),
            SizedBox(height: 20),

            /// 예시 이미지
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                _buildExampleItem(
                  imagePath: 'assets/crack_good_example.png',
                  label: 'Good',
                  labelColor: Colors.green,
                ),
                SizedBox(width: 20),
                _buildExampleItem(
                  imagePath: 'assets/crack_bad_example.png',
                  label: 'Bad',
                  labelColor: Colors.red,
                ),
              ],
            ),
            SizedBox(height: 30),

            /// 버튼
            Column(
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
            ),
            SizedBox(height: 30),

            /// 주의사항
            Text(
              '주의 사항\n• 본 진단은 AI로 진행하는 간이 검사이므로\n  정확한 균열 정도와는 다소 차이가 있을 수 있습니다.',
              style: TextStyle(color: Colors.grey[600], fontSize: 12),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }

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
}
