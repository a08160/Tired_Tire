import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'welcome_page.dart';

class ProfileEditPage extends StatefulWidget {
  const ProfileEditPage({Key? key}) : super(key: key);

  @override
  _ProfileEditPageState createState() => _ProfileEditPageState();
}

class _ProfileEditPageState extends State<ProfileEditPage> {
  late TextEditingController _nicknameController;
  bool _isChanged = false;
  bool _isSaving = false;
  File? _pickedImage;
  final ImagePicker _picker = ImagePicker();

  String _nickname = '';
  String _imageUrl = '';
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _nicknameController = TextEditingController();

    _nicknameController.addListener(() {
      setState(() {
        _isChanged =
            _nicknameController.text != _nickname || _pickedImage != null;
      });
    });

    _loadUserProfile();
  }

  Future<void> _loadUserProfile() async {
    try {
      final uid = FirebaseAuth.instance.currentUser!.uid;
      final userDoc =
          await FirebaseFirestore.instance.collection('users').doc(uid).get();

      if (userDoc.exists) {
        final data = userDoc.data()!;
        _nickname = data['nickname'] ?? '';
        _imageUrl = data['imageUrl'] ?? 'https://example.com/default_image.png';
        _nicknameController.text = _nickname;
      }
    } catch (e) {
      print('Error loading profile: $e');
    } finally {
      setState(() => _isLoading = false);
    }
  }

  Future<void> _pickImage() async {
    final XFile? pickedFile = await _picker.pickImage(
      source: ImageSource.gallery,
    );
    if (pickedFile != null) {
      setState(() {
        _pickedImage = File(pickedFile.path);
        _isChanged = true;
      });
    }
  }

  Future<void> _showCustomDialog({
    required String message,
    required String confirmText,
    required VoidCallback onConfirm,
  }) async {
    return showDialog(
      context: context,
      barrierDismissible: false,
      builder:
          (_) => Dialog(
            backgroundColor: Color(0xFFFFFFFF),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
            child: SizedBox(
              width: 240, // 가로 고정
              // height는 지정 안함 -> 자동조절
              child: Padding(
                padding: const EdgeInsets.symmetric(
                  horizontal: 20,
                  vertical: 20,
                ),
                child: Column(
                  mainAxisSize: MainAxisSize.min, // 높이 최소화해서 내용 크기에 맞춤
                  children: [
                    Text(
                      message,
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        color: Color(0xFF121212),
                        fontSize: 14,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                    SizedBox(height: 20),
                    Row(
                      children: [
                        Expanded(
                          child: ElevatedButton(
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Color(0xFFE8E8E8),
                              foregroundColor: Color(0xFF666666),
                              elevation: 0,
                              padding: EdgeInsets.symmetric(vertical: 10),
                            ),
                            onPressed: () => Navigator.pop(context),
                            child: Text(
                              "취소",
                              style: TextStyle(
                                fontSize: 14,
                                fontWeight: FontWeight.w500,
                              ),
                            ),
                          ),
                        ),
                        SizedBox(width: 12),
                        Expanded(
                          child: ElevatedButton(
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Color(0xFF121212),
                              foregroundColor: Colors.white,
                              elevation: 0,
                              padding: EdgeInsets.symmetric(vertical: 10),
                            ),
                            onPressed: () {
                              Navigator.pop(context);
                              onConfirm();
                            },
                            child: Text(
                              confirmText,
                              style: TextStyle(
                                fontSize: 14,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ),
    );
  }

  void _showSaveDialog() {
    _showCustomDialog(
      message: "프로필에 수정사항이 있습니다\n저장하시겠습니까?",
      confirmText: "저장",
      onConfirm: () async {
        setState(() => _isSaving = true);
        try {
          final uid = FirebaseAuth.instance.currentUser!.uid;

          await FirebaseFirestore.instance.collection('users').doc(uid).update({
            'nickname': _nicknameController.text,
          });

          setState(() {
            _nickname = _nicknameController.text;
            _isChanged = false;
          });

          ScaffoldMessenger.of(
            context,
          ).showSnackBar(SnackBar(content: Text('프로필이 저장되었습니다.')));
        } catch (e) {
          ScaffoldMessenger.of(
            context,
          ).showSnackBar(SnackBar(content: Text('저장 중 오류가 발생했습니다: $e')));
        } finally {
          setState(() => _isSaving = false);
        }
      },
    );
  }

  void _showLogoutDialog() {
    _showCustomDialog(
      message: "지금 로그아웃 하시나요?",
      confirmText: "로그아웃",
      onConfirm: () async {
        await FirebaseAuth.instance.signOut();
        Navigator.pushAndRemoveUntil(
          context,
          MaterialPageRoute(builder: (_) => WelcomePage()),
          (route) => false,
        );
      },
    );
  }

  void _showWithdrawDialog() {
    _showCustomDialog(
      message: "회원탈퇴를 하시나요?\n계정은 영구 삭제됩니다.",
      confirmText: "탈퇴",
      onConfirm: () async {
        final user = FirebaseAuth.instance.currentUser;
        if (user == null) return;

        try {
          final uid = user.uid;

          await FirebaseFirestore.instance
              .collection('users')
              .doc(uid)
              .delete();
          await user.delete();

          Navigator.pushAndRemoveUntil(
            context,
            MaterialPageRoute(builder: (_) => WelcomePage()),
            (route) => false,
          );

          ScaffoldMessenger.of(
            context,
          ).showSnackBar(SnackBar(content: Text("회원 탈퇴가 완료되었습니다.")));
        } on FirebaseAuthException catch (e) {
          if (e.code == 'requires-recent-login') {
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(
                content: Text("보안을 위해 최근 로그인 후 탈퇴할 수 있습니다. 다시 로그인 해주세요."),
              ),
            );
          } else {
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(content: Text("탈퇴 중 오류가 발생했습니다: ${e.message}")),
            );
          }
        } catch (e) {
          ScaffoldMessenger.of(
            context,
          ).showSnackBar(SnackBar(content: Text("탈퇴 중 오류가 발생했습니다.")));
        }
      },
    );
  }

  @override
  void dispose() {
    _nicknameController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        title: Text('프로필 수정', style: TextStyle(color: Colors.black)),
        backgroundColor: Colors.white,
        elevation: 0,
        leading: BackButton(color: Colors.black),
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            Stack(
              children: [
                CircleAvatar(
                  radius: 60,
                  backgroundImage:
                      _pickedImage != null
                          ? FileImage(_pickedImage!) as ImageProvider
                          : NetworkImage(_imageUrl),
                ),
                Positioned(
                  bottom: 0,
                  right: 0,
                  child: GestureDetector(
                    onTap: _pickImage,
                    child: CircleAvatar(
                      backgroundColor: Colors.deepPurple,
                      radius: 16,
                      child: Icon(
                        Icons.camera_alt,
                        color: Colors.white,
                        size: 16,
                      ),
                    ),
                  ),
                ),
              ],
            ),
            SizedBox(height: 24),
            Align(
              alignment: Alignment.centerLeft,
              child: Text("닉네임", style: TextStyle(fontWeight: FontWeight.bold)),
            ),
            SizedBox(height: 8),
            TextField(
              cursorColor: Colors.black,
              controller: _nicknameController,
              decoration: InputDecoration(
                hintText: "닉네임 입력",
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
                enabledBorder: OutlineInputBorder(
                  borderSide: BorderSide(color: Colors.black),
                ),
                focusedBorder: OutlineInputBorder(
                  borderSide: BorderSide(color: Colors.black),
                ),
              ),
            ),
            SizedBox(height: 24),
            if (_isChanged)
              ElevatedButton(
                onPressed: _showSaveDialog,
                style: ElevatedButton.styleFrom(
                  minimumSize: Size(double.infinity, 48),
                  backgroundColor: Colors.black,
                ),
                child: Text("변경 사항 저장"),
              ),
            Spacer(),
            TextButton(
              onPressed: _showLogoutDialog,
              child: Text("로그아웃", style: TextStyle(color: Colors.black)),
            ),
            TextButton(
              onPressed: _showWithdrawDialog,
              child: Text("회원탈퇴", style: TextStyle(color: Colors.red)),
            ),
          ],
        ),
      ),
    );
  }
}
